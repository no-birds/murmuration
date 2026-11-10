package org.nobirds.murmuration.agent

import org.nobirds.murmuration.prompt.*

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import org.nobirds.murmuration.internal.logger
import org.nobirds.murmuration.tooling.ToolCallingInput
import org.nobirds.murmuration.tooling.ToolOutcome
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger
import kotlin.time.Clock
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Instant

/**
 * A single execution entry reported by an agent, including name, duration and optional statistics.
 */
sealed interface AgentExecutionEntry {
    val agentName: String
    val duration: Duration

    /**
     * Composition path of the [step] frame this entry was reported from, outermost first
     * (`["SceneCycle", "Staging", "Cartographer"]`). Empty means the entry was reported straight
     * at the pipeline root — the flat, pre-step-frames shape, which is also what every legacy
     * record looks like.
     *
     * Agents never fill this themselves: the frame's context stamps it on the way out (see
     * `StepExecutionContext`), so an agent stays unaware of who composed it.
     */
    val stepPath: List<String> get() = emptyList()

    /** Returns a copy of this entry carrying [path]. Used by the frame to stamp its own path. */
    fun withStepPath(path: List<String>): AgentExecutionEntry
}

/**
 * Nature of the `PromptExecutionEntry` event:
 *
 * - [ROUND_TRIP] — a regular model call. Costs are charged as input/output at regular rates.
 *   If `statistics.cacheId != null`, an existing cache was used (and its share is
 *   visible in `tokens.input.cached`). This is the main case for the cost calculator.
 *
 * - [CACHE_CREATE] — creation of a Gemini Context Cache as a separate API call. `tokens.input.total`
 *   = size of the created cache; output = 0; `cacheId` = id of the created cache; `requestMessages`
 *   = full cache content (what was put into it). `LlmCostCalculator` excludes
 *   CACHE_CREATE entries from aggregation — the creation cost is accounted for via the
 *   `isCacheCreated` flag of the next ROUND_TRIP after the cache. CACHE_CREATE rows are needed for
 *   post-analysis of prompts, not for cost.
 */
enum class ExecutionNature { ROUND_TRIP, CACHE_CREATE }

data class PromptExecutionEntry(
    override val agentName: String,
    override val duration: Duration,
    val statistics: PromptExecutionStatistics,
    val nature: ExecutionNature = ExecutionNature.ROUND_TRIP,
    /**
     * Correlates this completed round-trip with its earlier [ExecutionContext.onAgentTurn] start,
     * so a durable store can write a RUNNING row at the start and UPDATE it here. Defaulted for
     * call sites that don't pair with an `onAgentTurn` (e.g. cache-create entries).
     */
    val executionId: UUID = UUID.randomUUID(),
    override val stepPath: List<String> = emptyList(),
    /**
     * The fact of every tool call resolved during this round-trip — see [org.nobirds.murmuration.tooling.ToolOutcome].
     * Empty for entries that carry no tool calls (plain-text turns, cache-create rows, non-tool-calling
     * agents). Lives here rather than inside [PromptExecutionStatistics]: that type describes prompt
     * execution proper (what was sent, what came back) and is the `@Serializable` shape mirrored into
     * `llm_statistics` — tool outcomes are not part of that mirror.
     */
    val toolOutcomes: List<ToolOutcome> = emptyList(),
) : AgentExecutionEntry {
    override fun withStepPath(path: List<String>): PromptExecutionEntry = copy(stepPath = path)
}

data class LambdaAgentExecutionEntry(
    override val agentName: String,
    override val duration: Duration,
    val success: Boolean,
    val errorMessage: String? = null,
    override val stepPath: List<String> = emptyList(),
) : AgentExecutionEntry {
    override fun withStepPath(path: List<String>): LambdaAgentExecutionEntry = copy(stepPath = path)
}

/**
 * Information about a single failed model invocation.
 */
data class ModelInvocationFailure(
    val agentName: String,
    val errorMessage: String?
)

/**
 * Aggregated token statistics for a single model.
 */
data class ModelTokenStatistics(
    val modelName: String,
    val totalTokens: Long,
    val input: InputTokenStatistics,
    val output: OutputTokenStatistics,
    val invocations: Int,
    val totalDuration: Duration,
    val failures: List<ModelInvocationFailure> = emptyList(),
    val cache: ModelCacheStatistics = ModelCacheStatistics(),
)

/**
 * Cache stats scoped to a single model. Mixing them across models in one bucket loses both
 * cost (different per-token prices) and ttl (different cache backends/scopes), so the
 * structure is per-model and only aggregated at the top level for display convenience.
 */
data class ModelCacheStatistics(
    val created: Int = 0,
    val createdTokens: Long = 0,
    val createdTtl: Duration? = null,
    val reused: Int = 0,
    val reusedTokens: Long = 0,
    val reusedTtl: Duration? = null,
    val expired: Int = 0,
)

/**
 * Aggregated pipeline statistics collected from all agents.
 *
 * @property iterations Number of full planner→implementor→verifier cycles reported by the
 *   pipeline through [ExecutionContext.reportIteration]. Stays at 0 for pipelines that don't
 *   wrap their work in such a cycle.
 * @property cacheLifetimeMillis Actual lived time (creation → deletion) for `PromptCacheScope.PIPELINE`
 *   caches reported via [PipelineEventListener.onCacheDeleted], keyed by cache id. Empty for caches
 *   not yet deleted at the time this snapshot was assembled (e.g. deletion races with report
 *   building — see [PipelineStatisticsCollector.onCacheDeleted]).
 * @property cacheEstimatedLifetimeMillis Elapsed-based lifetime *estimate* (pipeline-finish − cache-
 *   creation), keyed by cache id, filled at assembly for `PromptCacheScope.PIPELINE` caches whose
 *   exact deletion time hasn't arrived yet. Rationale: a PIPELINE-scope cache dies together with the
 *   pipeline, so once the pipeline has finished its real lifetime is ≈ (finish − creation) — orders
 *   of magnitude below the declared TTL (prod: ~22s vs 10min). Only present when the pipeline has
 *   finished and the cache is known to be pipeline-scoped; otherwise callers fall back to the
 *   declared TTL (an upper bound). Precedence for cost/labels: exact → estimate → TTL.
 */
data class PipelineStatistics(
    val executions: List<AgentExecutionEntry>,
    val byModel: Map<String, ModelTokenStatistics>,
    val iterations: Int = 0,
    val cacheLifetimeMillis: Map<String, Long> = emptyMap(),
    val cacheEstimatedLifetimeMillis: Map<String, Long> = emptyMap(),
) {
    /**
     * Top-level cache aggregate computed from per-model [ModelTokenStatistics.cache].
     * Kept as a `val`-property (not data-class field) so it stays consistent with [byModel]
     * automatically — there is no separate place to drift out of sync.
     *
     * Sums across models are honest for counts/tokens; [createdTtl]/[reusedTtl] pick the
     * last non-null value across models and are mostly diagnostic (the per-model TTL is
     * the authoritative one).
     */
    val cacheStats: CacheStatistics get() {
        val perModel = byModel.values.map { it.cache }
        if (perModel.all { it.created == 0 && it.reused == 0 && it.expired == 0 }) return CacheStatistics()
        return CacheStatistics(
            created = perModel.sumOf { it.created },
            createdTokens = perModel.sumOf { it.createdTokens },
            createdTtl = perModel.mapNotNull { it.createdTtl }.lastOrNull(),
            reused = perModel.sumOf { it.reused },
            reusedTokens = perModel.sumOf { it.reusedTokens },
            reusedTtl = perModel.mapNotNull { it.reusedTtl }.lastOrNull(),
            expired = perModel.sumOf { it.expired },
        )
    }

    data class CacheStatistics(
        val created: Int = 0,
        val createdTokens: Long = 0,
        val createdTtl: Duration? = null,
        val reused: Int = 0,
        val reusedTokens: Long = 0,
        val reusedTtl: Duration? = null,
        val expired: Int = 0
    )

    override fun toString(): String {
        val modelLines = byModel.values.joinToString("\n  ") { m ->
            val failuresInfo = if (m.failures.isNotEmpty()) ", failures=${m.failures.size}" else ""
            val cacheInfo = m.cache.renderInline()?.let { ", cache=$it" }.orEmpty()
            "${m.modelName}: ${m.invocations} calls in ${m.totalDuration}, " +
                "input=${m.input}, output=${m.output}, total=${m.totalTokens}$failuresInfo$cacheInfo"
        }
        return "PipelineStatistics(agents=${executions.size}, iterations=$iterations, models:\n  $modelLines)"
    }
}

private fun ModelCacheStatistics.renderInline(): String? {
    if (created == 0 && reused == 0 && expired == 0) return null
    val createdInfo = if (created > 0) "$created created ($createdTokens tok, ttl=$createdTtl)" else null
    val reusedInfo = if (reused > 0) "$reused reused ($reusedTokens tok, remaining ttl=$reusedTtl)" else null
    val expiredInfo = if (expired > 0) "$expired expired" else null
    return listOfNotNull(createdInfo, reusedInfo, expiredInfo).joinToString(", ")
}

/**
 * Execution context that agents use during execution.
 * Provides [reportAgentExecution] for reporting results; the context itself is a
 * [CoroutineScope], so coroutines are launched on it directly.
 */
interface ExecutionContext : CoroutineScope {
    /**
     * Unique identifier for the pipeline execution.
     */
    val pipelineId: UUID

    /**
     * Composition path of the [step] frame this context belongs to, outermost first. Empty at
     * the pipeline root; every [step] hands its block a child context with one more segment.
     * Read it to attribute work to a node of the tree — do not build one by hand.
     */
    val stepPath: List<String> get() = emptyList()

    /**
     * Reports that a [step] frame has opened. [stepId] correlates this start with the matching
     * [reportStepFinished] — the same start/finish correlation pattern as [onAgentTurn] and
     * [PromptExecutionEntry.executionId], so a durable store can insert a RUNNING row here and
     * UPDATE it on finish even when two frames share a path.
     */
    suspend fun reportStepStarted(stepId: UUID, path: List<String>) {}

    /**
     * Reports that a [step] frame has closed, successfully or not. [error] is the message of the
     * throwable that left the frame (`null` on success) — this is the channel through which a
     * failed deterministic sub-step, which produces no LLM round-trip at all, becomes visible.
     */
    suspend fun reportStepFinished(
        stepId: UUID,
        path: List<String>,
        status: StepStatus,
        duration: Duration,
        error: String?,
    ) {}

    /**
     * Reports an agent execution result. `suspend` because under the hood we invoke
     * [PipelineEventListener] — listener methods may write to async sinks (HTTP/DB/Kafka).
     */
    suspend fun reportAgentStarted(agentName: String)
    suspend fun reportAgentFinished(agentName: String)
    suspend fun reportAgentExecution(entry: AgentExecutionEntry)

    /**
     * Reports a retry attempt within a single logical agent run. Distinct from
     * [reportAgentStarted] — retries do **not** start a new logical agent (they are a
     * continuation of the same work, see [WrapperAgent]), so progress counters must
     * not increment. Use this for diagnostic UI ("retrying after ${cause::class}").
     */
    suspend fun reportAgentRetry(agentName: String, attempt: Int, cause: Throwable)

    /**
     * Reports that a wrapper layer was entered (Retry, Recovery, Adapter, …). The wrapper
     * is transparent to the START/END stream (its `name == delegate.name`), but admin
     * trace may want to know the wrapper chain. Optional — listeners default to no-op.
     */
    suspend fun reportWrapperLayer(layerKind: String, delegateName: String)

    /**
     * Reports the start of a new full pipeline iteration (for example, the next
     * planner→implementor→verifier cycle). Used to count the cycles in
     * [PipelineStatistics.iterations].
     */
    suspend fun reportIteration()

    /**
     * Display name for an agent with the given name, including the invocation suffix
     * (`agentName(N)`) when the agent has been started more than once in this run.
     */
    fun getDisplayName(agentName: String): String

    /**
     * Asks the runtime whether the next LLM round-trip is allowed. Read-only:
     * [PipelineRun] does NOT mutate counters here — it just delegates to its
     * [BudgetStrategy], which consults whichever [MetricsAccumulator] (or other
     * source) it was constructed with. Agents call this before every
     * `promptExecutor.execute(...)`.
     */
    fun beforeCall(): CallLimitDecision = CallLimitDecision.Proceed

    /**
     * Throwing variant of [beforeCall] for single-shot agents ([TextPromptAgent],
     * [ImageAgent]) that have no partial result to return. The default delegates to
     * [beforeCall] and throws [PipelineLimitException] (or rethrows the [Throwable] from
     * [CallLimitDecision.Throw]).
     */
    fun checkCallLimit() {
        when (val decision = beforeCall()) {
            is CallLimitDecision.Proceed -> Unit
            is CallLimitDecision.GracefulStop -> throw PipelineLimitException(decision.reason)
            is CallLimitDecision.Throw -> throw decision.cause
        }
    }

    // --- Lifecycle hooks. Each one fans the event out to all registered
    //     [PipelineEventListener]s; default no-op for stub contexts. ---

    /**
     * Called at the start of each round-trip an agent attempts (successful or not). [executionId]
     * correlates this start with the later [reportAgentExecution] for the same round-trip.
     */
    suspend fun onAgentTurn(executionId: UUID, agentName: String) {}

    /**
     * Called the moment this round-trip's stream produces its first event. See
     * [PipelineEventListener.onStreamStarted] for why the dispatch site matters.
     */
    suspend fun onStreamStarted(executionId: UUID, modelName: String) {}

    /**
     * Called after a successful `promptExecutor.execute(...)` returned a response.
     * [toolCallsInResponse] is the number of tool-call entries the model emitted in this
     * single response (0 for plain text responses).
     */
    suspend fun afterRequest(toolCallsInResponse: Int) {}

    /**
     * Called after each tool-call has been dispatched (for fine-grained per-tool stats).
     * [executionId] is the enclosing round-trip's correlation id — the same one
     * [onAgentTurn]/[reportAgentExecution] use — so a listener can count tool-call events per
     * round-trip and compare that count against the [PromptExecutionEntry.toolOutcomes] the
     * round-trip's entry eventually carries (this event is dispatched from a DIFFERENT code path
     * inside `executeCall`, independently of the outcome it produces).
     */
    suspend fun afterToolCall(toolName: String, executionId: UUID) {}

    /** Called when `promptExecutor.execute(...)` threw — the model returned nothing useful. */
    suspend fun onFailure(cause: Throwable) {}

    /** Called by `RetryAgent` for every retry attempt. */
    suspend fun onRetry(attempt: Int, total: Int, cause: Throwable) {}

    /**
     * Cancel the execution context and all coroutines within its scope.
     */
    fun cancel()

    /**
     * Registers a pipeline-level listener. Registered listeners receive events alongside the
     * one passed to [executePipeline]. Returns an [AutoCloseable] for unregistration — used
     * by the runtime to drop scoped subscriptions automatically (for example, for the
     * duration of a single agent call). Registration is safe under concurrent access.
     */
    fun register(listener: PipelineEventListener): AutoCloseable

    /**
     * Registers an agent-level listener. The base interface is a marker; to receive specific
     * events, implement one of the capability subinterfaces (for example,
     * `ToolCallingListener`). Each agent decides which capabilities it dispatches —
     * listeners silently miss events from agents that don't emit them.
     */
    fun register(listener: AgentEventListener): AutoCloseable

    /**
     * Fires a pipeline-level error to all registered [PipelineEventListener]s. Called by
     * the runtime when [executePipelineWithBudget] catches a throwable from the user block.
     */
    suspend fun fireError(cause: Throwable)

    /**
     * Iterates registered [AgentEventListener]s. Used by capability-typed dispatch helpers
     * (see `dispatchToolCalling` and similar) to fan events out without exposing the
     * underlying listener storage.
     */
    suspend fun forEachAgentListener(block: suspend (AgentEventListener) -> Unit)
}

/**
 * Reports a CACHE_CREATE row from a [PromptExecutionEvent.CacheCreated] event captured by an
 * agent's flow collector. Agent-name-aware helper: callers pass their own display name, and
 * this extension wraps the event into a [PromptExecutionEntry] with `nature = CACHE_CREATE`
 * (suffixed with `[cache_create]`) before reporting it. Used by every agent that may receive
 * a [PromptExecutionEvent.CacheCreated] event from the executor.
 */
suspend fun ExecutionContext.reportCacheCreate(
    agentDisplayName: String,
    event: PromptExecutionEvent.CacheCreated,
) {
    val statistics = PromptExecutionStatistics(
        modelName = event.modelName,
        success = true,
        tokens = event.tokens,
        cacheId = event.cacheId,
        cachedTokenCount = event.tokenCount,
        isCacheCreated = true,
        cacheTtlMillis = event.ttl?.inWholeMilliseconds,
        cacheScope = event.scope,
        requestMessages = event.requestMessages,
    )
    reportAgentExecution(
        PromptExecutionEntry(
            agentName = "$agentDisplayName[cache_create]",
            duration = event.duration,
            statistics = statistics,
            nature = ExecutionNature.CACHE_CREATE,
        )
    )
}

/**
 * Convenience extension: execute an agent within the current [ExecutionContext].
 *
 * This is the single chokepoint of agent invocation, so it is also where a composition node is
 * opened: the agent runs inside its own [step] frame ([String.toStepName] of the agent's name),
 * which makes the tree of invocations the tree of steps — no separate concept, no extra call
 * site to keep in sync. Wrapper agents (retry, recovery, adapters) call their delegate raw, so
 * they stay transparent here too: no second frame, no duplicated node.
 */
suspend fun <In, Out> ExecutionContext.execute(agent: Agent<In, Out>, input: In): Out {
    reportAgentStarted(agent.name)
    try {
        return step(agent.name.toStepName()) { with(agent) { execute(input) } }
    } finally {
        reportAgentFinished(agent.name)
    }
}

/**
 * Convenience extension: execute an agent that requires a tool within the current [ExecutionContext].
 */
suspend fun <In, Tool : Any, Out> ExecutionContext.execute(
    agent: Agent<ToolCallingInput<In, Tool>, Out>,
    input: In,
    tool: Tool
): Out = execute(agent, ToolCallingInput(input, tool))

/**
 * Convenience extension: schedule an agent execution within the current [ExecutionContext].
 */
fun <In, Out> ExecutionContext.scheduleExecute(agent: Agent<In, Out>, input: In): Deferred<Out> {
    return async {
        execute(agent, input)
    }
}


/**
 * Listener for pipeline execution events.
 */
interface PipelineEventListener {
    /**
     * Called when a pipeline execution starts.
     */
    suspend fun onStarted(pipelineId: UUID) {}

    /**
     * Called when an agent execution is reported.
     */
    suspend fun onEntryReported(pipelineId: UUID, entry: AgentExecutionEntry) {}

    /**
     * Called when an agent starts execution.
     */
    suspend fun onAgentStarted(pipelineId: UUID, agentName: String) {}

    /**
     * Called when an agent finishes execution.
     */
    suspend fun onAgentFinished(pipelineId: UUID, agentName: String) {}

    /**
     * Called on each retry attempt inside a single logical agent run. Diagnostic-only —
     * does **not** participate in the START/END count. See [ExecutionContext.reportAgentRetry].
     */
    suspend fun onAgentRetry(pipelineId: UUID, agentName: String, attempt: Int, cause: Throwable) {}

    /**
     * Called when a wrapper layer is entered. Diagnostic-only — listeners may use this
     * to build a composition tree for admin trace. See [ExecutionContext.reportWrapperLayer].
     */
    suspend fun onWrapperLayer(pipelineId: UUID, layerKind: String, delegateName: String) {}

    /**
     * Called when a [step] frame opens. [path] is the full composition path of the frame,
     * outermost first; [stepId] correlates with the matching [onStepFinished].
     */
    suspend fun onStepStarted(pipelineId: UUID, stepId: UUID, path: List<String>) {}

    /**
     * Called when a [step] frame closes. Carries the frame's outcome, wall time and — for a
     * frame that threw — the error message. Unlike [onEntryReported], this fires for steps with
     * no LLM round-trip at all, which is what makes a broken deterministic sub-step diagnosable.
     */
    suspend fun onStepFinished(
        pipelineId: UUID,
        stepId: UUID,
        path: List<String>,
        status: StepStatus,
        duration: Duration,
        error: String?,
    ) {}

    /**
     * Called when a pipeline execution fails.
     */
    suspend fun onError(pipelineId: UUID, throwable: Throwable) {}

    /**
     * Called when a pipeline reports a new full iteration (one full cycle of agents).
     */
    suspend fun onIteration(pipelineId: UUID) {}

    /**
     * Called at the start of each agent round-trip (one `promptExecutor.execute(...)`
     * attempt). Fires for every attempt, successful or not. [executionId] is the per-round-trip
     * correlation id, matching the [PromptExecutionEntry.executionId] reported on completion —
     * a durable writer inserts a RUNNING row here and UPDATEs it on [onEntryReported].
     */
    suspend fun onAgentTurn(pipelineId: UUID, executionId: UUID, agentName: String) {}

    /**
     * Called the moment the provider answers a round-trip (`StreamResponded`) —
     * parallel to [onAgentTurn], but fires even on a round-trip that later breaks mid-stream:
     * [onAgentTurn] opens the RUNNING record BEFORE the executor is even called, while this
     * fires only once the provider actually answered. [executionId] is the same round-trip
     * correlation id as [onAgentTurn]/[onEntryReported]. Dispatched from INSIDE the stream's
     * `collect` — not after `TurnResult` is
     * assembled, which an obryv-stream round-trip never reaches.
     */
    suspend fun onStreamStarted(pipelineId: UUID, executionId: UUID, modelName: String) {}

    /**
     * Called after a successful `promptExecutor.execute(...)` returned a response.
     * [toolCallsInResponse] is the number of tool-call entries the model emitted in this
     * single response (0 for plain text responses).
     */
    suspend fun onRequest(pipelineId: UUID, toolCallsInResponse: Int) {}

    /**
     * Called once per tool-call dispatched by the runtime. [executionId] is the enclosing
     * round-trip's correlation id, matching [onAgentTurn] and [PromptExecutionEntry.executionId] —
     * see [ExecutionContext.afterToolCall] for why this correlation exists.
     */
    suspend fun onToolCall(pipelineId: UUID, executionId: UUID, toolName: String) {}

    /** Called when `promptExecutor.execute(...)` threw — the model returned nothing useful. */
    suspend fun onFailure(pipelineId: UUID, cause: Throwable) {}

    /** Called by `RetryAgent` for every retry attempt (successful or not). */
    suspend fun onRetry(pipelineId: UUID, attempt: Int, total: Int, cause: Throwable) {}

    /**
     * Called when a pipeline execution finishes.
     */
    suspend fun onFinished(pipelineId: UUID) {}

    /**
     * Called by the cache subsystem after a previously created cache has been deleted
     * (`PromptCacheScope.PIPELINE` caches go away when the pipeline finishes; other scopes
     * may stay around). [lifetimeMillis] is how long the cache actually lived between
     * creation and deletion — used for accurate cache-storage billing in place of the
     * declared TTL (an upper bound). Default no-op; application-side writers override to UPDATE the
     * corresponding `llm_statistics` CACHE_CREATE row.
     */
    suspend fun onCacheDeleted(pipelineId: UUID, cacheId: String, lifetimeMillis: Long) {}

    companion object {
        /**
         * Creates a composite listener that delegates events to all provided listeners.
         */
        fun composite(vararg listeners: PipelineEventListener?): PipelineEventListener =
            CompositePipelineEventListener(listeners.filterNotNull())

        /**
         * Creates a composite listener that delegates events to all provided listeners.
         */
        fun composite(listeners: List<PipelineEventListener?>): PipelineEventListener =
            CompositePipelineEventListener(listeners.filterNotNull())
    }
}

/**
 * Composite listener that owns a mutable, thread-safe set of child listeners and fans events
 * out to each. Acts as the single point of registration/unregistration — [PipelineRun] holds
 * a reference and delegates `register(...)` here instead of maintaining its own list.
 *
 * @param initial Listeners registered at construction time (kept across the lifetime of this
 *   composite). They are dispatched in registration order, with subsequently-added listeners
 *   appended to the tail.
 */
class CompositePipelineEventListener(initial: List<PipelineEventListener> = emptyList()) : PipelineEventListener {
    private val listeners = java.util.concurrent.CopyOnWriteArrayList<PipelineEventListener>().apply { addAll(initial) }

    private val logger = logger<CompositePipelineEventListener>()

    /** Adds a listener and returns an [AutoCloseable] that removes it again. */
    fun register(listener: PipelineEventListener): AutoCloseable {
        listeners.add(listener)
        return AutoCloseable { listeners.remove(listener) }
    }

    override suspend fun onStarted(pipelineId: UUID) {
        listeners.forEach { it.onStarted(pipelineId) }
    }

    override suspend fun onEntryReported(pipelineId: UUID, entry: AgentExecutionEntry) {
        listeners.forEach { it.onEntryReported(pipelineId, entry) }
    }

    override suspend fun onAgentStarted(pipelineId: UUID, agentName: String) {
        listeners.forEach { it.onAgentStarted(pipelineId, agentName) }
    }

    override suspend fun onAgentFinished(pipelineId: UUID, agentName: String) {
        listeners.forEach { it.onAgentFinished(pipelineId, agentName) }
    }

    override suspend fun onAgentRetry(pipelineId: UUID, agentName: String, attempt: Int, cause: Throwable) {
        listeners.forEach { it.onAgentRetry(pipelineId, agentName, attempt, cause) }
    }

    override suspend fun onWrapperLayer(pipelineId: UUID, layerKind: String, delegateName: String) {
        listeners.forEach { it.onWrapperLayer(pipelineId, layerKind, delegateName) }
    }

    /**
     * Step frames are the only events fanned out with per-listener isolation. The sibling
     * methods above deliberately stay bare (a listener that throws there is a bug worth
     * surfacing), but a step frame is *paired*: a store writes RUNNING on start and only the
     * finish report moves it to a terminal state. One listener throwing would take the finish
     * report away from every listener registered after it — turning "listener B is broken" into
     * "half the tree is stuck RUNNING", which reads as a crashed pod.
     */
    override suspend fun onStepStarted(pipelineId: UUID, stepId: UUID, path: List<String>) {
        listeners.forEach {
            try {
                it.onStepStarted(pipelineId, stepId, path)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                logger.warn("Step listener ${it::class.simpleName} failed on start of $path (pipeline $pipelineId)", e)
            }
        }
    }

    override suspend fun onStepFinished(
        pipelineId: UUID,
        stepId: UUID,
        path: List<String>,
        status: StepStatus,
        duration: Duration,
        error: String?,
    ) {
        listeners.forEach {
            try {
                it.onStepFinished(pipelineId, stepId, path, status, duration, error)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                logger.warn("Step listener ${it::class.simpleName} failed on finish of $path (pipeline $pipelineId)", e)
            }
        }
    }

    override suspend fun onError(pipelineId: UUID, throwable: Throwable) {
        listeners.forEach { it.onError(pipelineId, throwable) }
    }

    override suspend fun onIteration(pipelineId: UUID) {
        listeners.forEach { it.onIteration(pipelineId) }
    }

    override suspend fun onAgentTurn(pipelineId: UUID, executionId: UUID, agentName: String) {
        listeners.forEach { it.onAgentTurn(pipelineId, executionId, agentName) }
    }

    override suspend fun onStreamStarted(pipelineId: UUID, executionId: UUID, modelName: String) {
        listeners.forEach { it.onStreamStarted(pipelineId, executionId, modelName) }
    }

    override suspend fun onRequest(pipelineId: UUID, toolCallsInResponse: Int) {
        listeners.forEach { it.onRequest(pipelineId, toolCallsInResponse) }
    }

    override suspend fun onToolCall(pipelineId: UUID, executionId: UUID, toolName: String) {
        listeners.forEach { it.onToolCall(pipelineId, executionId, toolName) }
    }

    override suspend fun onFailure(pipelineId: UUID, cause: Throwable) {
        listeners.forEach { it.onFailure(pipelineId, cause) }
    }

    override suspend fun onRetry(pipelineId: UUID, attempt: Int, total: Int, cause: Throwable) {
        listeners.forEach { it.onRetry(pipelineId, attempt, total, cause) }
    }

    override suspend fun onFinished(pipelineId: UUID) {
        listeners.forEach { it.onFinished(pipelineId) }
    }

    override suspend fun onCacheDeleted(pipelineId: UUID, cacheId: String, lifetimeMillis: Long) {
        listeners.forEach { it.onCacheDeleted(pipelineId, cacheId, lifetimeMillis) }
    }
}

/**
 * Thread-safe registry for [AgentEventListener]s. Mirrors [CompositePipelineEventListener] but
 * for agent-level listeners — agents iterate [snapshot] and dispatch via capability filters
 * (see `dispatchToolCalling` and similar). Owned by [PipelineRun]; exposed read-only to agents.
 */
class AgentEventListenerRegistry(initial: List<AgentEventListener> = emptyList()) {
    private val listeners = java.util.concurrent.CopyOnWriteArrayList<AgentEventListener>().apply { addAll(initial) }

    fun register(listener: AgentEventListener): AutoCloseable {
        listeners.add(listener)
        return AutoCloseable { listeners.remove(listener) }
    }

    /** Read-only view; iteration is safe under concurrent mutation (`CopyOnWriteArrayList`). */
    val snapshot: List<AgentEventListener> get() = listeners
}

/**
 * Marker interface for all agent-level listeners. By itself it carries no handlers — to
 * receive events, implement a capability subinterface (for example, `ToolCallingListener`
 * for round-trips and tool calls).
 *
 * Registration goes through [ExecutionContext.register]. Each agent decides which
 * capability it dispatches (`dispatchToolCalling` and similar). Listeners silently miss
 * events from agents that don't emit the matching capability — this is by design.
 *
 * A tool/builder may implement a listener interface on itself — the runtime auto-subscribes
 * it for the duration of a single agent call (see `executeWithTools` auto-subscribe in
 * `tooling`).
 */
interface AgentEventListener

/**
 * A collector that gathers execution entries via [PipelineEventListener] and provides [PipelineStatistics].
 */
class PipelineStatisticsCollector : PipelineEventListener {
    private val entries = ConcurrentLinkedQueue<AgentExecutionEntry>()
    private val iterations = AtomicInteger(0)

    // Actual cache lifetimes, keyed by cacheId, as they arrive from CachePipelineCleaner. One
    // collector instance is scoped to a single pipeline run (see call sites), so no need to
    // filter by pipelineId here — mirrors onEntryReported's own lack of filtering.
    private val cacheLifetimes = ConcurrentHashMap<String, Long>()

    // Observed creation time per cache (when its CACHE_CREATE entry was reported) and the pipeline's
    // finish time — the two ends of the elapsed-based lifetime estimate for PIPELINE-scope caches
    // (see [PipelineStatistics.cacheEstimatedLifetimeMillis]). The estimate exists because deletion
    // is async and its exact lifetime routinely arrives too late for the synchronous report build.
    private val cacheCreatedAt = ConcurrentHashMap<String, Instant>()
    @Volatile private var finishedAt: Instant? = null

    override suspend fun onEntryReported(pipelineId: UUID, entry: AgentExecutionEntry) {
        entries.add(entry)
        if (entry is PromptExecutionEntry && entry.nature == ExecutionNature.CACHE_CREATE) {
            entry.statistics.cacheId?.let { cacheCreatedAt.putIfAbsent(it, Clock.System.now()) }
        }
    }

    /**
     * Records pipeline finish time — the far end of the elapsed lifetime estimate. Both the
     * success ([onFinished]) and failure ([onError]) paths stamp it: a PIPELINE-scope cache is
     * cleaned up on either (`CachePipelineCleaner` hooks both), so "when did the pipeline stop"
     * is the right anchor regardless of outcome.
     */
    override suspend fun onFinished(pipelineId: UUID) {
        if (finishedAt == null) finishedAt = Clock.System.now()
    }

    override suspend fun onError(pipelineId: UUID, throwable: Throwable) {
        if (finishedAt == null) finishedAt = Clock.System.now()
    }

    override suspend fun onIteration(pipelineId: UUID) {
        iterations.incrementAndGet()
    }

    /**
     * Records the actual lifetime of a deleted `PIPELINE`-scope cache — see
     * [PipelineStatistics.cacheLifetimeMillis]. Note this event is emitted from a fire-and-forget
     * coroutine launched by `CachePipelineCleaner.onFinished` (which itself performs a real
     * `backend.deleteCache` network round-trip), racing against whatever reads
     * [PipelineStatisticsCollector.statistics] right after the pipeline call returns — it may well
     * arrive too late for that first read. Cost callers must treat a missing entry as "unknown",
     * not "cache lived forever" — see [LlmCostCalculator.perModel]'s TTL fallback.
     */
    override suspend fun onCacheDeleted(pipelineId: UUID, cacheId: String, lifetimeMillis: Long) {
        cacheLifetimes[cacheId] = lifetimeMillis
    }

    /**
     * Returns the aggregated statistics from all collected agent executions, grouped by model name.
     */
    val statistics: PipelineStatistics
        get() = assemble()

    private fun assemble(): PipelineStatistics {
        val all = entries.toList()
        val byModel = all
            .filterIsInstance<PromptExecutionEntry>()
            .groupBy { it.statistics.modelName }
            .mapValues { (modelName, modelEntries) ->
                val created = modelEntries.filter { it.nature == ExecutionNature.CACHE_CREATE }
                // Reuse: regular round-trips that hit an existing cache (cacheId != null).
                // cached_tokens — the share of input actually taken from the cache in this row.
                val reused = modelEntries.filter {
                    it.nature == ExecutionNature.ROUND_TRIP && it.statistics.cacheId != null
                }
                ModelTokenStatistics(
                    modelName = modelName,
                    totalTokens = modelEntries.sumOf { it.statistics.tokens.totalTokens },
                    input = InputTokenStatistics(
                        total = modelEntries.sumOf { it.statistics.tokens.input.total },
                        cached = modelEntries.sumOf { it.statistics.tokens.input.cached },
                        cacheWriteShort = modelEntries.sumOf { it.statistics.tokens.input.cacheWriteShort },
                        cacheWriteLong = modelEntries.sumOf { it.statistics.tokens.input.cacheWriteLong }
                    ),
                    output = OutputTokenStatistics(
                        total = modelEntries.sumOf { it.statistics.tokens.output.total },
                        thinking = modelEntries.sumOf { it.statistics.tokens.output.thinking }
                    ),
                    invocations = modelEntries.size,
                    totalDuration = modelEntries.fold(Duration.ZERO) { acc, e -> acc + e.duration },
                    failures = modelEntries
                        .filter { !it.statistics.success }
                        .map { ModelInvocationFailure(it.agentName, it.statistics.errorMessage) },
                    cache = ModelCacheStatistics(
                        created = created.size,
                        createdTokens = created.sumOf { it.statistics.tokens.input.total },
                        createdTtl = created.mapNotNull { it.statistics.cacheTtlMillis?.milliseconds }.lastOrNull(),
                        reused = reused.size,
                        reusedTokens = reused.sumOf { it.statistics.tokens.input.cached },
                        reusedTtl = reused.mapNotNull { it.statistics.cacheTtlMillis?.milliseconds }.lastOrNull(),
                        expired = modelEntries.count { it.statistics.isCacheExpired == true },
                    ),
                )
            }

        // Elapsed lifetime estimate for PIPELINE-scope caches whose exact deletion time hasn't
        // arrived: once the pipeline has finished, the cache is gone with it, so (finish − creation)
        // is a far tighter figure than the declared TTL. Skip caches with a known exact lifetime
        // (that always wins) and non-PIPELINE scopes (they outlive the pipeline — TTL stays the
        // honest upper bound).
        val estimatedLifetimes: Map<String, Long> = finishedAt?.let { finish ->
            all.asSequence()
                .filterIsInstance<PromptExecutionEntry>()
                .filter { it.nature == ExecutionNature.CACHE_CREATE && it.statistics.cacheScope == PromptCacheScope.PIPELINE }
                .mapNotNull { e ->
                    val cacheId = e.statistics.cacheId ?: return@mapNotNull null
                    if (cacheLifetimes.containsKey(cacheId)) return@mapNotNull null
                    val createdAt = cacheCreatedAt[cacheId] ?: return@mapNotNull null
                    cacheId to (finish - createdAt).inWholeMilliseconds.coerceAtLeast(0)
                }
                .toMap()
        } ?: emptyMap()

        return PipelineStatistics(
            all,
            byModel,
            iterations = iterations.get(),
            cacheLifetimeMillis = cacheLifetimes.toMap(),
            cacheEstimatedLifetimeMillis = estimatedLifetimes,
        )
    }
}

/**
 * A pipeline run that manages execution lifecycle and dispatches events.
 * Created per pipeline invocation.
 */
class PipelineRun(
    override val pipelineId: UUID = UUID.randomUUID(),
    parentScope: CoroutineScope,
    initialEventListener: PipelineEventListener? = null,
    /**
     * Decider consulted on every `beforeCall`. The runtime passes a snapshot of the internal
     * accumulator; whatever configuration the strategy needs (e.g. [BudgetLimits] for
     * [DefaultBudgetStrategy]) is its own concern, set up at construction time.
     */
    private val budgetStrategy: BudgetStrategy = DefaultBudgetStrategy(),
) : ExecutionContext {

    private val job = SupervisorJob(parentScope.coroutineContext[Job])
    override val coroutineContext = parentScope.coroutineContext + job

    // Internal accumulator: the budget strategy's source of truth on every `beforeCall`,
    // and the synchronous counter for `getDisplayName`. Strictly an implementation detail —
    // callers who want their own metrics register a separate `MetricsAccumulator` (or any
    // listener) via the constructor argument or `register(...)`.
    private val builtInMetrics = MetricsAccumulator()

    // Two parallel registries — one per kind of listener. Both own storage, add/remove and
    // dispatch (composite fans events out; agent registry exposes iteration). PipelineRun
    // doesn't manage any list itself, just delegates `register(...)` here. Built-in
    // accumulator is added first so its counters are up-to-date by the time the budget
    // strategy is consulted on the next `beforeCall`.
    private val pipelineListeners = CompositePipelineEventListener(
        listOfNotNull(builtInMetrics, initialEventListener)
    )
    private val agentListeners = AgentEventListenerRegistry()

    override fun register(listener: PipelineEventListener): AutoCloseable = pipelineListeners.register(listener)
    override fun register(listener: AgentEventListener): AutoCloseable = agentListeners.register(listener)

    override suspend fun fireError(cause: Throwable) {
        pipelineListeners.onError(pipelineId, cause)
    }

    override suspend fun forEachAgentListener(block: suspend (AgentEventListener) -> Unit) {
        for (listener in agentListeners.snapshot) block(listener)
    }

    override suspend fun reportAgentExecution(entry: AgentExecutionEntry) {
        pipelineListeners.onEntryReported(pipelineId, entry)
    }

    override suspend fun reportIteration() {
        pipelineListeners.onIteration(pipelineId)
    }

    override suspend fun reportAgentStarted(agentName: String) {
        // Increment synchronously before the event so the display name reflects this
        // invocation. Counts live in the accumulator (single source of truth).
        builtInMetrics.recordAgentInvocation(agentName)
        pipelineListeners.onAgentStarted(pipelineId, getDisplayName(agentName))
    }

    override suspend fun reportAgentFinished(agentName: String) {
        pipelineListeners.onAgentFinished(pipelineId, getDisplayName(agentName))
    }

    override suspend fun reportAgentRetry(agentName: String, attempt: Int, cause: Throwable) {
        pipelineListeners.onAgentRetry(pipelineId, getDisplayName(agentName), attempt, cause)
    }

    override suspend fun reportWrapperLayer(layerKind: String, delegateName: String) {
        pipelineListeners.onWrapperLayer(pipelineId, layerKind, getDisplayName(delegateName))
    }

    override suspend fun reportStepStarted(stepId: UUID, path: List<String>) {
        pipelineListeners.onStepStarted(pipelineId, stepId, path)
    }

    override suspend fun reportStepFinished(
        stepId: UUID,
        path: List<String>,
        status: StepStatus,
        duration: Duration,
        error: String?,
    ) {
        pipelineListeners.onStepFinished(pipelineId, stepId, path, status, duration, error)
    }

    override fun getDisplayName(agentName: String): String {
        val count = builtInMetrics.agentInvocationCount(agentName)
        return if (count > 1) agentName.withIteration(count) else agentName
    }

    override fun beforeCall(): CallLimitDecision = budgetStrategy.decide(builtInMetrics.snapshot())

    override suspend fun onAgentTurn(executionId: UUID, agentName: String) {
        pipelineListeners.onAgentTurn(pipelineId, executionId, agentName)
    }
    override suspend fun onStreamStarted(executionId: UUID, modelName: String) {
        pipelineListeners.onStreamStarted(pipelineId, executionId, modelName)
    }
    override suspend fun afterRequest(toolCallsInResponse: Int) {
        pipelineListeners.onRequest(pipelineId, toolCallsInResponse)
    }
    override suspend fun afterToolCall(toolName: String, executionId: UUID) {
        pipelineListeners.onToolCall(pipelineId, executionId, toolName)
    }
    override suspend fun onFailure(cause: Throwable) {
        pipelineListeners.onFailure(pipelineId, cause)
    }
    override suspend fun onRetry(attempt: Int, total: Int, cause: Throwable) {
        pipelineListeners.onRetry(pipelineId, attempt, total, cause)
    }

    suspend fun start() {
        pipelineListeners.onStarted(pipelineId)
    }

    suspend fun finish() {
        pipelineListeners.onFinished(pipelineId)
    }

    override fun cancel() {
        job.cancel()
    }
}

/**
 * Legacy entry point: takes the old [maxCalls] number and a [CallLimitStrategy]. Wraps the
 * legacy strategy into a [BudgetStrategy] that inspects [ExecutionMetrics.requests] — the
 * count of successful LLM round-trips (`promptExecutor.execute(...)` calls), which is what
 * [beforeCall] is actually invoked once per (see its KDoc) and what the strategies' own
 * messages claim to limit ("... LLM calls"). Do NOT swap this for [ExecutionMetrics.toolCalls]:
 * that counter is the number of tool invocations across all round-trips, which models that
 * batch many tool-calls into a single response inflate far faster than actual LLM calls —
 * that mismatch previously caused pipelines to hit the "150 LLM calls" limit after only
 * ~20 real LLM round-trips.
 * Prefer [executePipelineWithBudget] for new code.
 */
suspend fun <T> CoroutineScope.executePipeline(
    pipelineId: UUID = UUID.randomUUID(),
    pipelineEventListener: PipelineEventListener? = null,
    maxCalls: Int = 50,
    callLimitStrategy: CallLimitStrategy = DefaultCallLimitStrategy,
    block: suspend ExecutionContext.() -> T,
): T {
    val strategy = BudgetStrategy { metrics ->
        callLimitStrategy.decide(metrics.requests, maxCalls)
    }
    return executePipelineWithBudget(
        pipelineId = pipelineId,
        pipelineEventListener = pipelineEventListener,
        budgetStrategy = strategy,
        block = block,
    )
}

/**
 * Full-budget entry point: pass a [BudgetStrategy] (with whatever config it needs baked in).
 * The runtime keeps an internal [MetricsAccumulator] that the strategy consults; callers who
 * want their own metrics in UI/logs simply register a separate accumulator (or any
 * [PipelineEventListener]) via [pipelineEventListener] — they're independent.
 */
suspend fun <T> CoroutineScope.executePipelineWithBudget(
    pipelineId: UUID = UUID.randomUUID(),
    pipelineEventListener: PipelineEventListener? = null,
    budgetStrategy: BudgetStrategy = DefaultBudgetStrategy(),
    block: suspend ExecutionContext.() -> T,
): T {
    val executionContext = PipelineRun(pipelineId, this, pipelineEventListener, budgetStrategy)
    var caught: Throwable? = null
    try {
        executionContext.start()
        with(executionContext) {
            return block()
        }
    } catch (e: Throwable) {
        caught = e
        throw e
    } finally {
        if (caught != null) executionContext.fireError(caught)
        else executionContext.finish()
        executionContext.cancel()
    }
}
