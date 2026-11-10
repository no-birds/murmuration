package org.nobirds.murmuration.agent

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * Live snapshot of pipeline-execution counters. Exposed by [ExecutionContext.metrics] and
 * passed to [BudgetStrategy] so strategies can decide based on the full picture
 * (iterations, requests, tool-calls, failures, retries) rather than a single number.
 *
 * All fields are cumulative for the lifetime of the pipeline run.
 */
data class ExecutionMetrics(
    /** Number of full outer-loop cycles ([ExecutionContext.reportIteration]). */
    val outerIterations: Int = 0,
    /**
     * Total LLM round-trips: every successful `promptExecutor.execute(...)` increments this.
     * A failed execute (timeout, network error — model returned nothing) does NOT count;
     * see [failures] for those.
     */
    val requests: Int = 0,
    /** Tool-calls actually issued by the model across all successful responses. */
    val toolCalls: Int = 0,
    /** Per-agent round-trip count (successful turns only). */
    val perAgentTurns: Map<String, Int> = emptyMap(),
    /**
     * Per-agent invocation count — how many times an agent with that name was started
     * via `ExecutionContext.execute(agent, ...)`. Drives the `agentName(N)` display
     * suffix when an agent is invoked more than once in the same pipeline run.
     */
    val agentInvocations: Map<String, Int> = emptyMap(),
    /**
     * Failed `execute(...)` invocations — execution threw before the model returned a
     * useful response (idle timeout, connection drop, 5xx etc.). Pure waste, does not
     * touch [requests]/[toolCalls].
     */
    val failures: Int = 0,
    /** Retry attempts performed by `RetryAgent` across the run. */
    val retries: Int = 0,
)

/**
 * [PipelineEventListener] that turns the event stream into a live [ExecutionMetrics]
 * snapshot. Reusable: any consumer that wants metrics (UI live view, statistics dump,
 * a [BudgetStrategy]) creates its own instance and registers it with the runtime — they
 * don't share state.
 */
class MetricsAccumulator : PipelineEventListener {
    private val outerIterations = AtomicInteger(0)
    private val requests = AtomicInteger(0)
    private val toolCalls = AtomicInteger(0)
    private val failures = AtomicInteger(0)
    private val retries = AtomicInteger(0)
    private val perAgentTurns = ConcurrentHashMap<String, AtomicInteger>()
    private val agentInvocations = ConcurrentHashMap<String, AtomicInteger>()

    override suspend fun onIteration(pipelineId: java.util.UUID) {
        outerIterations.incrementAndGet()
    }

    /**
     * Direct, synchronous increment of the per-agent invocation counter. Called by
     * [PipelineRun.reportAgentStarted] *before* the `onAgentStarted` event is dispatched
     * so that listeners (and `getDisplayName`) see the up-to-date count.
     *
     * Not driven from [onAgentStarted] because that event carries the *display* name
     * (`agentName(N)`) for UI consumers, which is the wrong key for counting.
     */
    fun recordAgentInvocation(agentName: String) {
        agentInvocations.computeIfAbsent(agentName) { AtomicInteger(0) }.incrementAndGet()
    }

    override suspend fun onAgentTurn(pipelineId: java.util.UUID, executionId: java.util.UUID, agentName: String) {
        perAgentTurns.computeIfAbsent(agentName) { AtomicInteger(0) }.incrementAndGet()
    }

    override suspend fun onRequest(pipelineId: java.util.UUID, toolCallsInResponse: Int) {
        requests.incrementAndGet()
        if (toolCallsInResponse > 0) toolCalls.addAndGet(toolCallsInResponse)
    }

    override suspend fun onFailure(pipelineId: java.util.UUID, cause: Throwable) {
        failures.incrementAndGet()
    }

    override suspend fun onRetry(pipelineId: java.util.UUID, attempt: Int, total: Int, cause: Throwable) {
        retries.incrementAndGet()
    }

    /**
     * Read-only fast path: how many times an agent with that name was invoked.
     * Used by `PipelineRun.getDisplayName` to compute the `(N)` suffix synchronously.
     */
    fun agentInvocationCount(agentName: String): Int =
        agentInvocations[agentName]?.get() ?: 0

    fun snapshot(): ExecutionMetrics = ExecutionMetrics(
        outerIterations = outerIterations.get(),
        requests = requests.get(),
        toolCalls = toolCalls.get(),
        perAgentTurns = perAgentTurns.mapValues { it.value.get() },
        agentInvocations = agentInvocations.mapValues { it.value.get() },
        failures = failures.get(),
        retries = retries.get(),
    )
}

/**
 * Pure-function decider: given a snapshot of [ExecutionMetrics] decides whether the pipeline
 * keeps going. Stateless w.r.t. accumulators — the runtime passes a fresh snapshot on every
 * call. Whatever configuration a strategy needs (limits, weights, schedules) is its own
 * concern, set up at construction time.
 */
fun interface BudgetStrategy {
    fun decide(metrics: ExecutionMetrics): CallLimitDecision
}

/**
 * Configuration for [DefaultBudgetStrategy] — soft/hard limits per metric.
 * Defaults are conservative; concrete pipelines tune them at strategy-construction time.
 */
data class BudgetLimits(
    /** Outer-loop cap (planner → implementors → verifier cycles). */
    val maxOuterIterations: Int = 3,
    /** Per-agent local round-trip cap; agents enforce their own `maxTurns` already. */
    val maxAgentTurnsPerAgent: Int = 20,
    /**
     * Pipeline-wide budget on tool-calls — the main "real work" metric. A pipeline that
     * has produced this many tool invocations is asked to stop gracefully.
     */
    val maxToolCalls: Int = 50,
    /**
     * Defensive ceiling on total LLM round-trips. Catches degenerate cases where the
     * model returns text without tool-calls round after round; not the day-to-day budget.
     */
    val maxRequests: Int = 200,
    /** Hard stop after this many cumulative failures. */
    val maxFailures: Int = 10,
    /** Hard stop after this many retry attempts across all agents. */
    val maxRetries: Int = 6,
)

/**
 * Default strategy: stop gracefully on the first limit exceeded. Limits are part of the
 * strategy's own config — not the runtime's — so different strategies can carry different
 * shapes of configuration.
 */
class DefaultBudgetStrategy(val limits: BudgetLimits = BudgetLimits()) : BudgetStrategy {
    override fun decide(metrics: ExecutionMetrics): CallLimitDecision = when {
        metrics.toolCalls >= limits.maxToolCalls ->
            CallLimitDecision.GracefulStop("tool-call budget exhausted (${metrics.toolCalls}/${limits.maxToolCalls})")
        metrics.requests >= limits.maxRequests ->
            CallLimitDecision.GracefulStop("request ceiling reached (${metrics.requests}/${limits.maxRequests})")
        metrics.failures >= limits.maxFailures ->
            CallLimitDecision.GracefulStop("too many failures (${metrics.failures}/${limits.maxFailures})")
        metrics.retries >= limits.maxRetries ->
            CallLimitDecision.GracefulStop("retry ceiling reached (${metrics.retries}/${limits.maxRetries})")
        metrics.outerIterations >= limits.maxOuterIterations ->
            CallLimitDecision.GracefulStop("outer-iteration cap reached (${metrics.outerIterations}/${limits.maxOuterIterations})")
        else -> CallLimitDecision.Proceed
    }
}
