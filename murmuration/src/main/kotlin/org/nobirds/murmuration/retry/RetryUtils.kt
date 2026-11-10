package org.nobirds.murmuration.retry

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.tooling.AgentAbortException

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.withTimeout
import org.slf4j.LoggerFactory
import java.util.Collections
import java.util.IdentityHashMap
import kotlin.time.Duration
import kotlin.time.Duration.Companion.minutes

/**
 * Shortcut: tests whether an exception is retryable under [DefaultRetryStrategy] (core only,
 * no providers). Application code that uses provider-specific strategies typically composes
 * its own helper, e.g. `Throwable.isAppRetryable()` over
 * `DefaultRetryStrategy or GeminiRetryStrategy`.
 */
fun Throwable.isDefaultRetryable(): Boolean = DefaultRetryStrategy.shouldRetry(this, 0)

/**
 * This exception followed by its cause chain. Cycles (`A -> B -> A`, possible through `initCause`)
 * end at the first repeated throwable: the walk compares by identity, so a cycle cannot loop forever
 * and every distinct throwable is yielded once. Each iteration gets its own visited set.
 */
val Throwable.causes: Sequence<Throwable>
    get() = sequence {
        val visited = Collections.newSetFromMap(IdentityHashMap<Throwable, Boolean>())
        var current: Throwable? = this@causes
        while (current != null && visited.add(current)) {
            yield(current)
            current = current.cause
        }
    }

/**
 * True when an [AgentAbortException] sits anywhere in this exception's cause chain (including
 * wrapped by a foreign exception). Such a failure is terminal for every framework retry: the
 * decision points ([RetrySession.nextDelayOrNull], [retryWithTimeout]) check it BEFORE the
 * [RetryStrategy], so strategies need not know about aborts.
 */
val Throwable.abortsAgent: Boolean
    get() = causes.any { it is AgentAbortException }

/**
 * Rethrows this exception UNCHANGED (wrapper and cause chain intact) when it [abortsAgent]; returns
 * normally otherwise. The first line of every `catch` in the agentic loop that would otherwise turn
 * the exception into a tool result, a WARN or a continuation.
 */
fun Throwable.rethrowIfAborts() {
    if (abortsAgent) throw this
}

private val retryWithTimeoutLogger = LoggerFactory.getLogger("RetryWithTimeout")

/**
 * Runs [block] with a per-attempt timeout and retries. Use this for non-agent operations
 * (cache create/refresh, side-channel API calls) where `RetryAgent` is not applicable.
 *
 * A thin wrapper over [RetrySession]: it owns only the per-attempt `withTimeout`. Pauses, budgets
 * (quota vs. everything else), the quota ladder, the server retry-after hint and the strategy
 * verdict all come from [policy] — the same behaviour as the round-trip retry, no second policy.
 *
 * The attempt timeout is mapped to [TransientException] with [TransientReason.NETWORK_TIMEOUT] and is
 * retried regardless of the strategy verdict (the wrapper decides it, the budget still applies).
 * External cancellation propagates without a retry; an [AgentAbortException] in the cause chain is
 * terminal. When the budget is exhausted the last exception propagates.
 */
suspend fun <T> retryWithTimeout(
    timeoutPerAttempt: Duration,
    policy: RetryPolicy = RetryPolicy(maxBackoff = 1.minutes),
    operationName: String = "operation",
    block: suspend () -> T
): T {
    val session = policy.newSession()
    while (true) {
        var timedOut = false
        val error: Throwable = try {
            return withTimeout(timeoutPerAttempt) { block() }
        } catch (e: TimeoutCancellationException) {
            timedOut = true
            TransientException(
                TransientReason.NETWORK_TIMEOUT,
                "$operationName timed out after $timeoutPerAttempt",
                null,
            )
        } catch (e: CancellationException) {
            // External cancellation, not our timeout — propagate without retrying.
            throw e
        } catch (e: Throwable) {
            e
        }
        val delayDuration = session.nextDelayOrNull(error, bypassStrategy = timedOut) ?: throw error
        retryWithTimeoutLogger.warn(
            "$operationName failed (retry ${session.attemptInBudget}/${policy.maxRetries}). Retrying in $delayDuration. Error: ${error.message}"
        )
        delay(delayDuration)
    }
}
