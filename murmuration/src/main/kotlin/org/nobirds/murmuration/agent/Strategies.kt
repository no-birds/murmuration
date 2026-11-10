package org.nobirds.murmuration.agent

import org.nobirds.murmuration.prompt.*
import org.nobirds.murmuration.llm.*
import org.nobirds.murmuration.retry.*

/**
 * Decision returned by [CallLimitStrategy] for the next LLM call.
 *
 * - [Proceed]: the call is allowed; the agent continues.
 * - [GracefulStop]: budget exhausted. Loop agents (`ToolCallingAgent`) exit with a partial
 *   result and emit `onCallLimitExhausted`. Single-shot agents ([TextPromptAgent],
 *   [ImageAgent]) — which have no partial result to return — throw [PipelineLimitException].
 * - [Throw]: the runtime must immediately throw the given exception (single-shot agents
 *   included).
 */
sealed interface CallLimitDecision {
    data object Proceed : CallLimitDecision
    data class GracefulStop(val reason: String) : CallLimitDecision
    data class Throw(val cause: Throwable) : CallLimitDecision
}

/**
 * SPI deciding whether the next LLM call is allowed. The runtime invokes it via
 * [ExecutionContext.beforeCall] before every prompt execution.
 *
 * Default is [DefaultCallLimitStrategy] (graceful stop on overflow), matching the existing
 * `maxTurns` semantics: warn and return a partial result. For strict fail-fast, use
 * [ThrowingCallLimitStrategy].
 */
fun interface CallLimitStrategy {
    fun decide(currentTotal: Int, maxCalls: Int): CallLimitDecision
}

/**
 * When [maxCalls] is exceeded, returns [CallLimitDecision.GracefulStop]: a `ToolCallingAgent`
 * exits the loop with a partial result and emits `onCallLimitExhausted`; single-shot agents
 * throw [PipelineLimitException].
 */
object DefaultCallLimitStrategy : CallLimitStrategy {
    override fun decide(currentTotal: Int, maxCalls: Int): CallLimitDecision =
        if (currentTotal > maxCalls)
            CallLimitDecision.GracefulStop("Pipeline reached global limit of $maxCalls LLM calls")
        else CallLimitDecision.Proceed
}

/**
 * Strict policy: when [maxCalls] is exceeded, returns [CallLimitDecision.Throw] with a
 * [PipelineLimitException]. Useful when the pipeline wants to fail fast instead of returning
 * a partial result.
 */
object ThrowingCallLimitStrategy : CallLimitStrategy {
    override fun decide(currentTotal: Int, maxCalls: Int): CallLimitDecision =
        if (currentTotal > maxCalls)
            CallLimitDecision.Throw(PipelineLimitException("Pipeline reached global limit of $maxCalls LLM calls"))
        else CallLimitDecision.Proceed
}
