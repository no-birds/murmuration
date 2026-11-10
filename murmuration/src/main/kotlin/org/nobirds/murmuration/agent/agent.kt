package org.nobirds.murmuration.agent

import org.nobirds.murmuration.prompt.*
import org.nobirds.murmuration.llm.*
import org.nobirds.murmuration.retry.*

import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.withContext
import org.nobirds.murmuration.internal.logger
import kotlin.reflect.KClass
import kotlin.time.TimeSource
import kotlin.time.measureTimedValue

interface Agent<In, Out> {
    val name: String get() = this::class.simpleName ?: "UnnamedAgent"

    /**
     * Execute this agent within the given [ExecutionContext].
     * Statistics and timing can be reported to the context.
     */
    suspend fun ExecutionContext.execute(input: In): Out
}

/**
 * Agent that performs the actual work — invokes an LLM, generates an image, runs a
 * tool-loop, executes a lambda. The `execute(agent, input)` extension fires exactly one
 * `onAgentStarted` / `onAgentFinished` pair per worker run (see AgentScope.kt). Counting
 * progress in pipelines is based on these events — one worker = one logical unit.
 */
interface WorkerAgent<In, Out> : Agent<In, Out>

/**
 * Composition agent that wraps another agent (`delegate`) — retry, recovery, adapter,
 * billing, etc. Wrappers must satisfy two contracts to keep progress counting honest:
 *
 *   1. `name == delegate.name` — wrappers are *transparent* to the event stream. The
 *      `onAgentStarted` / `onAgentFinished` pair fired by the outer extension carries
 *      the real worker's name, not "Retry(...)". This way a `toolAgent().withRetry()`
 *      registers as a single event for `BaseAgent`, not as `Retry(BaseAgent)` + `BaseAgent`.
 *
 *   2. Inside `execute()` the delegate is invoked via `with(delegate) { execute(input) }`
 *      (raw call) — **never** through the `execute(agent, input)` extension — so the same
 *      START/END pair isn't fired again.
 *
 * For traceability the wrapper *may* call [ExecutionContext.reportWrapperLayer] at the
 * start of its `execute()` (admin trace will see the composition tree without affecting
 * the progress counter), and [ExecutionContext.reportAgentRetry] on each retry attempt.
 */
interface WrapperAgent<In, Out> : Agent<In, Out> {
    val delegate: Agent<*, *>
}


internal fun generateAgentName(name: String, type: String): String {
    val id = java.util.UUID.randomUUID().toString().substring(0, 4)
    return "$name#$id[$type]"
}

fun String.withIteration(iteration: Int): String {
    val bracketIndex = indexOf('[')
    return if (bracketIndex != -1) {
        substring(0, bracketIndex) + "#$iteration" + substring(bracketIndex)
    } else {
        "$this#$iteration"
    }
}

/**
 * Insert a `[state]` segment into an agent name right before the existing `[type]` suffix,
 * e.g. `Agent#abcd#1[CallToolAgent]` + `withState("PLANNING")` →
 * `Agent#abcd#1[PLANNING][CallToolAgent]`. Used by [org.nobirds.murmuration.tooling.ToolCallingAgent]
 * to surface the current state-machine phase in agent names that end up in logs and DB
 * `agent_name` columns.
 */
fun String.withState(state: Any): String {
    val bracketIndex = indexOf('[')
    return if (bracketIndex != -1) {
        substring(0, bracketIndex) + "[$state]" + substring(bracketIndex)
    } else {
        "$this[$state]"
    }
}

class TextPromptAgent<In, Out>(
    private val promptExecutor: PromptExecutor,
    private val resultSerializer: TextPromptSerializer<Out>,
    private val promptGenerator: PromptGenerator<In>,
    name: String = "TextPromptAgent"
) : WorkerAgent<In, Out> {

    override val name: String = generateAgentName(name, "TextAgent")

    private val logger = logger()

    override suspend fun ExecutionContext.execute(input: In): Out {
        val displayName = getDisplayName(name)
        logger.info("Agent {} started", displayName)

        val startTime = TimeSource.Monotonic.markNow()
        try {
            val prompt = with(promptGenerator) { generate(input) }
            prompt.requireModalities(Modality.Text::class)
            checkCallLimit()
            val ctx = this
            // Publish this invocation's execution identity (pipelineId + agent displayName) into
            // the coroutine context for executors that support identity-aware request handling
            // (no-op on providers without one) — see [AgentExecutionIdentity] KDoc. Must wrap the
            // actual `.collect` (here, inside `.aggregate()`) — wrapping prompt construction
            // alone would have no effect.
            val result = withContext(AgentExecutionIdentity(pipelineId, displayName, stepPath)) {
                promptExecutor.execute(prompt)
                    .onEach { event ->
                        if (event is PromptExecutionEvent.CacheCreated) ctx.reportCacheCreate(displayName, event)
                    }
                    .aggregate()
            }
            val output = resultSerializer.serialize(result)
            val duration = startTime.elapsedNow()

            reportAgentExecution(PromptExecutionEntry(displayName, duration, result.statistics()))

            logger.info("Agent {} finished in {}", displayName, duration)
            if (logger.isDebugEnabled) logger.debug("Agent {} output: {}", displayName, output)

            return output
        } catch (e: Exception) {
            val duration = startTime.elapsedNow()
            val typed = e as? PromptExecutionException
            // Partial statistics already assembled before the failure (tokens, responseMessage,
            // model name — whatever `aggregate` carried through the exception) so the FAILURE
            // row reflects the round-trip's real partial content instead of an empty stub
            //. Null `partial` → default empty statistics, same as before.
            val partial = typed?.partialStats
            val stats = (partial ?: PromptExecutionStatistics()).copy(
                success = false,
                errorMessage = e.message,
                attempts = 1,
                errorCode = typed?.errorCode(),
                // `?:` — `finishReasonOrNull()`/`streamDiagnostics`
                // are non-null only for a `PolicyViolationException`; on any other exception type
                // this would otherwise clobber a value `partial` already carried.
                finishReason = typed?.finishReasonOrNull() ?: partial?.finishReason,
                streamDiagnostics = typed?.streamDiagnostics ?: partial?.streamDiagnostics,
            )
            reportAgentExecution(PromptExecutionEntry(displayName, duration, stats))
            logger.warn("Agent {} failed in {}. Error: {}", displayName, duration, e.message)
            throw e
        }
    }
}

/**
 * An agent that embeds its prompt's text into a dense vector. The prompt carries
 * [EmbeddingPromptSettings] (attached by the [embeddingAgent] factory); the shared
 * [PromptExecutor] sees the signal and routes the request to `embedContent`. Statistics
 * surface through the same path as text/image agents — the embedding call lands in
 * `llm_statistics` with its model name and usage.
 */
class EmbeddingPromptAgent<In>(
    private val promptExecutor: PromptExecutor,
    private val promptGenerator: PromptGenerator<In>,
    name: String = "EmbeddingPromptAgent",
) : WorkerAgent<In, FloatArray> {

    override val name: String = generateAgentName(name, "EmbeddingAgent")
    private val logger = logger()

    override suspend fun ExecutionContext.execute(input: In): FloatArray {
        val displayName = getDisplayName(name)
        logger.info("Agent {} started", displayName)

        val startTime = TimeSource.Monotonic.markNow()
        try {
            val prompt = with(promptGenerator) { generate(input) }
            prompt.requireModalities(Modality.Embedding::class)
            checkCallLimit()
            val ctx = this
            val result = promptExecutor.execute(prompt)
                .onEach { event ->
                    if (event is PromptExecutionEvent.CacheCreated) ctx.reportCacheCreate(displayName, event)
                }
                .aggregate()
            val output = (result as? EmbeddingPromptExecutionResult)?.embedding
                ?: throw PromptExecutorException(
                    "Expected EmbeddingPromptExecutionResult, got ${result::class.simpleName}"
                )
            val duration = startTime.elapsedNow()

            reportAgentExecution(PromptExecutionEntry(displayName, duration, result.statistics()))

            logger.info("Agent {} finished in {} (dim={})", displayName, duration, output.size)
            return output
        } catch (e: Exception) {
            val duration = startTime.elapsedNow()
            val typed = e as? PromptExecutionException
            // Partial statistics already assembled before the failure (tokens, responseMessage,
            // model name — whatever `aggregate` carried through the exception) so the FAILURE
            // row reflects the round-trip's real partial content instead of an empty stub
            //. Null `partial` → default empty statistics, same as before.
            val partial = typed?.partialStats
            val stats = (partial ?: PromptExecutionStatistics()).copy(
                success = false,
                errorMessage = e.message,
                attempts = 1,
                errorCode = typed?.errorCode(),
                // `?:` — see the identical comment above.
                finishReason = typed?.finishReasonOrNull() ?: partial?.finishReason,
                streamDiagnostics = typed?.streamDiagnostics ?: partial?.streamDiagnostics,
            )
            reportAgentExecution(PromptExecutionEntry(displayName, duration, stats))
            logger.warn("Agent {} failed in {}. Error: {}", displayName, duration, e.message)
            throw e
        }
    }
}

/**
 * An agent that executes a simple suspend block.
 */
class LambdaAgent<In, Out>(
    name: String = "LambdaAgent",
    private val isWrapper: Boolean = false,
    private val block: suspend ExecutionContext.(In) -> Out
) : WorkerAgent<In, Out> {

    override val name: String = if (isWrapper) name else generateAgentName(name, "Lambda")
    private val logger = logger()

    override suspend fun ExecutionContext.execute(input: In): Out {
        val displayName = getDisplayName(name)
        logger.info("Agent {} started", displayName)

        val startTime = TimeSource.Monotonic.markNow()
        try {
            val (output, time) = measureTimedValue {
                block(input)
            }

            reportAgentExecution(LambdaAgentExecutionEntry(displayName, time, true))

            logger.info("Agent {} finished in {}", displayName, time)
            if (logger.isDebugEnabled) logger.debug("Agent {} output: {}", displayName, output)

            return output
        } catch (e: Exception) {
            val duration = startTime.elapsedNow()
            reportAgentExecution(LambdaAgentExecutionEntry(displayName, duration, false, e.message))
            logger.warn("Agent {} failed in {}. Error: {}", displayName, duration, e.message)
            throw e
        }
    }
}

/**
 * Base agent for binary-media generation (image, music, …): builds the prompt via
 * [promptGenerator], runs it through the [PromptExecutor], aggregates the binary result, reports
 * statistics, and maps it to an [ImageAgentResponse] — [ImageAgentResponse.File] when a
 * `BinaryDataStorage` persisted the bytes, else in-memory [ImageAgentResponse.Bytes]. Modality
 * agents specialise it by the settings their prompt carries (image vs music) and by [typeName]
 * (the `[...]` tag in the agent name / `llm_statistics.agent_name`).
 *
 * [generateName]: when `false`, [name] is used as-is (an outer adapter already formatted it).
 */
open class MediaPromptAgent<In>(
    private val executor: PromptExecutor,
    private val promptGenerator: PromptGenerator<In>,
    private val outputModality: KClass<out Modality.Output>,
    name: String = "MediaPromptAgent",
    typeName: String = "MediaAgent",
    generateName: Boolean = true,
) : WorkerAgent<In, ImageAgentResponse> {

    final override val name: String = if (generateName) generateAgentName(name, typeName) else name
    private val logger = logger()

    final override suspend fun ExecutionContext.execute(input: In): ImageAgentResponse {
        val displayName = getDisplayName(name)
        logger.info("Agent {} started", displayName)

        val startTime = TimeSource.Monotonic.markNow()
        try {
            val prompt = with(promptGenerator) { generate(input) }
            prompt.requireModalities(outputModality)
            checkCallLimit()
            val ctx = this
            val result = executor.execute(prompt)
                .onEach { event ->
                    if (event is PromptExecutionEvent.CacheCreated) ctx.reportCacheCreate(displayName, event)
                }
                .aggregate() as? ImagePromptExecutionResult
                ?: throw PromptExecutorException(
                    "Media agent expected ImagePromptExecutionResult, but executor returned a different type"
                )
            val duration = startTime.elapsedNow()

            reportAgentExecution(PromptExecutionEntry(displayName, duration, result.statistics))

            val response: ImageAgentResponse = when (result) {
                is ImagePromptExecutionResult.File -> ImageAgentResponse.File(result.id, result.uri, result.mimeType)
                is ImagePromptExecutionResult.Bytes -> {
                    logger.warn(
                        "Agent {} produced Bytes-result (no BinaryDataStorage in the chain); " +
                            "downstream that needs a file URI will degrade to in-memory bytes.",
                        displayName,
                    )
                    ImageAgentResponse.Bytes(result.id, result.imageBytes, result.mimeType)
                }
            }

            logger.info("Agent {} finished in {}", displayName, duration)
            if (logger.isDebugEnabled) logger.debug("Agent {} output: {}", displayName, response)

            return response
        } catch (e: Exception) {
            val duration = startTime.elapsedNow()
            val typed = e as? PromptExecutionException
            // Partial statistics already assembled before the failure (tokens, responseMessage,
            // model name — whatever `aggregate` carried through the exception) so the FAILURE
            // row reflects the round-trip's real partial content instead of an empty stub
            //. Null `partial` → default empty statistics, same as before.
            val partial = typed?.partialStats
            val stats = (partial ?: PromptExecutionStatistics()).copy(
                success = false,
                errorMessage = e.message,
                attempts = 1,
                errorCode = typed?.errorCode(),
                // `?:` — see the identical comment above.
                finishReason = typed?.finishReasonOrNull() ?: partial?.finishReason,
                streamDiagnostics = typed?.streamDiagnostics ?: partial?.streamDiagnostics,
            )
            reportAgentExecution(PromptExecutionEntry(displayName, duration, stats))
            logger.warn("Agent {} failed in {}. Error: {}", displayName, duration, e.message)
            throw e
        }
    }
}

/**
 * Image generation: a [MediaPromptAgent] that attaches an optional [ImageSize] to the prompt
 * ([Prompt.withImageSize], stored in `ImagePromptSettings`) before running. Two flavours:
 * high-level (domain `In` + [promptGenerator] + [sizeCalculator]) and low-level (`In = Prompt`,
 * identity generator, no size — used inside [PromptRecoveryAgent]).
 */
class ImageAgent<In>(
    executor: PromptExecutor,
    promptGenerator: PromptGenerator<In>,
    sizeCalculator: (In) -> ImageSize? = { null },
    name: String = "ImagePromptAgent",
    generateName: Boolean = true,
) : MediaPromptAgent<In>(
    executor = executor,
    promptGenerator = promptGenerator.applyingImageSize(sizeCalculator),
    outputModality = Modality.Image::class,
    name = name,
    typeName = "ImageAgent",
    generateName = generateName,
)

/** Wraps a [PromptGenerator] so each generated prompt gets [ImageSize] applied (when the calculator returns one). */
private fun <In> PromptGenerator<In>.applyingImageSize(
    sizeCalculator: (In) -> ImageSize?,
): PromptGenerator<In> {
    val inner = this
    return PromptGenerator { input ->
        val base = with(inner) { generate(input) }
        sizeCalculator(input)?.let { base.withImageSize(it) } ?: base
    }
}

/** WrapperAgent that maps the output of [delegate] through [transform]. Transparent to
 *  the event stream — inherits delegate.name. */
private class MapAgent<In, Out, NewOut>(
    override val delegate: Agent<In, Out>,
    private val transform: (ExecutionContext, Out) -> NewOut,
) : WrapperAgent<In, NewOut> {
    override val name: String get() = delegate.name
    override suspend fun ExecutionContext.execute(input: In): NewOut {
        val raw = with(delegate) { execute(input) }
        return transform(this, raw)
    }
}

private class MapInputAgent<In, NewIn, Out>(
    override val delegate: Agent<NewIn, Out>,
    private val transform: (ExecutionContext, In) -> NewIn,
) : WrapperAgent<In, Out> {
    override val name: String get() = delegate.name
    override suspend fun ExecutionContext.execute(input: In): Out {
        val mapped = transform(this, input)
        return with(delegate) { execute(mapped) }
    }
}

fun <In, Out, NewOut> Agent<In, Out>.map(transform: (ExecutionContext, Out) -> NewOut): Agent<In, NewOut> =
    MapAgent(this, transform)

fun <In, NewIn, Out> Agent<NewIn, Out>.mapInput(transform: (ExecutionContext, In) -> NewIn): Agent<In, Out> =
    MapInputAgent(this, transform)
