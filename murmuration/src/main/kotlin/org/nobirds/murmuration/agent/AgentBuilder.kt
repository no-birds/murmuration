package org.nobirds.murmuration.agent

import org.nobirds.murmuration.prompt.*
import org.nobirds.murmuration.llm.*
import org.nobirds.murmuration.retry.*

import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient
import kotlinx.serialization.serializer

import org.nobirds.murmuration.internal.UUIDSerializer
import java.util.UUID

/**
 * Result of an image-generating agent. Mirrors [ImagePromptExecutionResult]:
 * a [Bytes] response still carries inline binary in memory (no storage in the chain);
 * a [File] response has already been persisted by a `BinaryDataStorage` and only
 * references the file by [File.uri]. Consumers should prefer the [File] form — for
 * downstream agents that need the image as input, pass the URI via `fileUri(...)` instead
 * of re-uploading bytes. Treat a [Bytes] result as «storage isn't wired in this
 * environment» (e.g. a unit test or a misconfiguration) — log loudly and fall back if you
 * still have a use for the raw blob.
 */
sealed interface ImageAgentResponse {
    val id: UUID
    val mimeType: String

    @Serializable
    data class Bytes(
        @Serializable(with = UUIDSerializer::class)
        override val id: UUID,
        @Transient
        val bytes: ByteArray = byteArrayOf(),
        override val mimeType: String,
    ) : ImageAgentResponse {
        override fun toString(): String =
            "ImageAgentResponse.Bytes(id=$id, bytes=${bytes.size}, mimeType=$mimeType)"
    }

    @Serializable
    data class File(
        @Serializable(with = UUIDSerializer::class)
        override val id: UUID,
        val uri: String,
        override val mimeType: String,
    ) : ImageAgentResponse
}

/**
 * Creates a simple LambdaAgent.
 */
inline fun <In, Out> agent(name: String = "LambdaAgent", crossinline block: suspend ExecutionContext.(In) -> Out): Agent<In, Out> =
    LambdaAgent(name, block = { block(it) })

/**
 * Creates a DSL-based builder for a Text Agent.
 */
inline fun <In, Out> textAgent(
    executor: PromptExecutor,
    serializer: TextPromptSerializer<Out>,
    name: String = "TextPromptAgent",
    crossinline block: suspend PromptBuilder.(ExecutionContext, In) -> Unit
): Agent<In, Out> = TextPromptAgent(executor, serializer, PromptGenerator.build { ec, input -> block(ec, input) }, name)

/**
 * Creates a DSL-based builder for a Text Agent that automatically handles JSON schema and deserialization.
 */
inline fun <In, reified Out : Any> textAgent(
    llm: Llm,
    executor: PromptExecutor,
    name: String = "${Out::class.simpleName}Agent",
    crossinline block: suspend PromptBuilder.(ExecutionContext, In) -> Unit
): Agent<In, Out> = textAgent(
    executor,
    TextPromptSerializer.Json(serializer<Out>()),
    name
) { ec, input ->
    this.llm = llm
    this.settings = TypedPromptSettings(Out::class)
    block(ec, input)
}

/**
 * Creates a DSL-based builder for an Image Agent.
 *
 * If [rewriter] is set, on a [PolicyViolationException] from Imagen the image prompt
 * goes through the rewriter (it receives the exception, reads the category and the
 * failed prompt from it, and returns a rewritten one), and is then retried up to [maxRewrites] times.
 * Recovery works at the Prompt level — one rewriter is shared between all image agents
 * of the project; category-aware logic (different instructions for PROHIBITED vs SAFETY) lives
 * inside the rewriter via `instructionFor`.
 */
inline fun <In> imageAgent(
    llm: Llm,
    executor: PromptExecutor,
    size: ImageSize = ImageSize.DEFAULT,
    rewriter: Agent<PolicyViolationException, Prompt>? = null,
    maxRewrites: Int = 1,
    name: String = "ImagePromptAgent",
    crossinline block: suspend PromptBuilder.(ExecutionContext, In) -> Unit
): Agent<In, ImageAgentResponse> = imageAgentInternal(
    executor,
    PromptGenerator.build(llm, block),
    { size },
    rewriter,
    maxRewrites,
    name,
)

/**
 * Creates a DSL-based builder for an Image Agent with dynamic size.
 *
 * See [imageAgent] above — the [rewriter] parameter works the same way.
 */
inline fun <In> imageAgent(
    llm: Llm,
    executor: PromptExecutor,
    rewriter: Agent<PolicyViolationException, Prompt>? = null,
    maxRewrites: Int = 1,
    name: String = "ImagePromptAgent",
    noinline sizeCalculator: (In) -> ImageSize?,
    crossinline block: suspend PromptBuilder.(ExecutionContext, In) -> Unit
): Agent<In, ImageAgentResponse> = imageAgentInternal(
    executor,
    PromptGenerator.build(llm, block),
    sizeCalculator,
    rewriter,
    maxRewrites,
    name,
)

/**
 * Creates a DSL-based builder for an Embedding Agent. The [block] builds the text to embed;
 * the factory attaches [EmbeddingPromptSettings] so the shared executor routes the prompt to
 * `embedContent`. Returns the dense vector; statistics/cost flow through the common pipe.
 */
inline fun <In> embeddingAgent(
    llm: Llm,
    executor: PromptExecutor,
    outputDimensionality: Int = 768,
    taskType: String = "SEMANTIC_SIMILARITY",
    name: String = "EmbeddingPromptAgent",
    crossinline block: suspend PromptBuilder.(ExecutionContext, In) -> Unit,
): Agent<In, FloatArray> = EmbeddingPromptAgent(
    executor,
    PromptGenerator.build { ec, input ->
        this.llm = llm
        this.settings = EmbeddingPromptSettings(outputDimensionality, taskType)
        block(ec, input)
    },
    name,
)

/**
 * Creates a DSL-based builder for a Music Agent. The [block] builds the prompt text; the factory
 * attaches [MusicPromptSettings] so the shared executor routes the request to music generation
 * (Lyria). Reuses [MediaPromptAgent] — statistics/cost flow through the common pipe like image.
 */
inline fun <In> musicAgent(
    llm: Llm,
    executor: PromptExecutor,
    name: String = "MusicPromptAgent",
    crossinline block: suspend PromptBuilder.(ExecutionContext, In) -> Unit,
): Agent<In, ImageAgentResponse> = MediaPromptAgent(
    executor = executor,
    promptGenerator = PromptGenerator.build { ec, input ->
        this.llm = llm
        this.settings = MusicPromptSettings()
        block(ec, input)
    },
    outputModality = Modality.Audio::class,
    name = name,
    typeName = "MusicAgent",
)

@PublishedApi
internal fun <In> imageAgentInternal(
    executor: PromptExecutor,
    promptGenerator: PromptGenerator<In>,
    sizeCalculator: (In) -> ImageSize?,
    rewriter: Agent<PolicyViolationException, Prompt>?,
    maxRewrites: Int,
    name: String,
): Agent<In, ImageAgentResponse> {
    if (rewriter == null) {
        return ImageAgent(executor, promptGenerator, sizeCalculator, name)
    }
    val fullName = generateAgentName(name, "ImageAgent")
    // Recovery operates on full Prompt — rewriter rebuilds the prompt on PolicyViolation.
    // Core is the same ImageAgent class with an identity generator and no size attachment
    // (the size has been applied earlier by the adapter, by the time the prompt reaches
    // here it already carries ImagePromptSettings via withImageSize).
    val core: Agent<Prompt, ImageAgentResponse> = ImageAgent(
        executor = executor,
        promptGenerator = PromptGenerator { it },
        sizeCalculator = { null },
        name = fullName,
        generateName = false,
    )
    val recovered: Agent<Prompt, ImageAgentResponse> = PromptRecoveryAgent(
        primary = core,
        recovery = rewriter,
        recoverableType = PolicyViolationException::class,
        maxRecoveries = maxRewrites,
    )
    return ImageAgentAdapter(fullName, recovered, promptGenerator, sizeCalculator)
}

/**
 * Thin wrapper: turns domain-`In` into a `Prompt` (via [promptGenerator] + size) and
 * delegates to the already-prepared image agent. Does not report its own execution-entry —
 * the statistics surface from [core] under the same user-facing [name] so logs and
 * `llm_statistics.agent_name` look identical to a flat `ImageAgent<In>`, despite the
 * `Recovery(ImageAgent)` chain under the hood.
 */
@PublishedApi
internal class ImageAgentAdapter<In>(
    override val name: String,
    private val core: Agent<Prompt, ImageAgentResponse>,
    private val promptGenerator: PromptGenerator<In>,
    private val sizeCalculator: (In) -> ImageSize?,
) : WrapperAgent<In, ImageAgentResponse> {

    override val delegate: Agent<*, *> get() = core

    override suspend fun ExecutionContext.execute(input: In): ImageAgentResponse {
        reportWrapperLayer("ImageAdapter", core.name)
        val basePrompt = with(promptGenerator) { generate(input) }
        val prompt = basePrompt.withImageSize(sizeCalculator(input) ?: ImageSize.DEFAULT)
        // Raw call — outer extension already fired START/END for adapter.name (== core.name
        // since core was built with `generateName = false, name = fullName`).
        return with(core) { execute(prompt) }
    }
}
