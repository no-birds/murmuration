package org.nobirds.murmuration.agent

import org.nobirds.murmuration.retry.*

import kotlinx.coroutines.delay
import org.nobirds.murmuration.internal.logger
import kotlin.time.Duration
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

/**
 * Retries the whole `delegate.execute(input)` on retryable failures. Attempt budget, backoff
 * and the retry decision come from a [RetryPolicy] (one [RetrySession] per `execute` run).
 *
 * Note the granularity: a retry here replays the **entire agent invocation** — for a
 * multi-round-trip tool agent that means rebuilding the conversation from scratch. For
 * cheap per-round-trip recovery inside the tool loop see `ToolCallingAgent`'s
 * `roundTripRetry`; this wrapper remains the outer safety net.
 */
class RetryAgent<In, Out>(
    override val delegate: Agent<In, Out>,
    private val policy: RetryPolicy,
) : WrapperAgent<In, Out> {

    constructor(
        delegate: Agent<In, Out>,
        maxRetries: Int = 3,
        initialBackoff: Duration = 2.seconds,
        maxBackoff: Duration = 10.minutes,
        retryStrategy: RetryStrategy = RetryStrategy { t, _ -> t.isDefaultRetryable() },
    ) : this(delegate, RetryPolicy(retryStrategy, maxRetries, initialBackoff, maxBackoff))

    // Wrapper is transparent to the event stream — it inherits delegate.name. A retry is a
    // continuation of the same logical agent run, not a new agent. See `WrapperAgent` kdoc.
    override val name: String get() = delegate.name

    private val logger = logger()

    override suspend fun ExecutionContext.execute(input: In): Out {
        reportWrapperLayer("Retry", delegate.name)
        val session = policy.newSession()

        while (true) {
            try {
                // Raw delegate invocation — `with(delegate) { execute(input) }` deliberately
                // bypasses the `execute(agent, input)` extension so the START/END pair is
                // fired exactly once by the outer extension that originally called us.
                return with(delegate) { execute(input) }
            } catch (e: Exception) {
                val delayDuration = session.nextDelayOrNull(e) ?: throw e

                reportAgentRetry(delegate.name, session.attempt, e)
                logger.warn(
                    "Agent {} failed (retry ${session.attemptInBudget}/${policy.maxRetries}). Retrying in $delayDuration. Error: ${e.message}",
                    delegate.name,
                )
                delay(delayDuration)
            }
        }
    }
}

fun <In, Out> Agent<In, Out>.withRetry(
    maxRetries: Int = 3,
    initialBackoff: Duration = 2.seconds,
    maxBackoff: Duration = 10.minutes,
    retryStrategy: RetryStrategy = RetryStrategy { t, _ -> t.isDefaultRetryable() }
): Agent<In, Out> = RetryAgent(this, maxRetries, initialBackoff, maxBackoff, retryStrategy)
