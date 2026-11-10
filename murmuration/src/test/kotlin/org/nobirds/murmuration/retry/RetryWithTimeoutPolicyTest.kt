package org.nobirds.murmuration.retry

import kotlinx.coroutines.delay
import kotlinx.coroutines.test.currentTime
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.QuotaExceededException
import org.nobirds.murmuration.prompt.QuotaKind
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds

/**
 * [retryWithTimeout] follows the SAME retry policy as the round-trip retry
 * ([RetrySession]) — quota ladder without a server hint, separate budgets per error class — and keeps
 * its own semantics: the per-attempt timeout is retried regardless of the strategy.
 *
 * Virtual time (`runTest`): the pauses are measured through `currentTime`, nothing really sleeps.
 */
class RetryWithTimeoutPolicyTest {

    private fun quota() = QuotaExceededException(QuotaKind.RATE_LIMIT, retryAfter = null, message = "429 RESOURCE_EXHAUSTED", cause = null)

    private fun other() = TransientException(TransientReason.SERVER_ERROR, "503", null)

    @Test
    fun `quota without a server hint sleeps on the quota ladder, not a flat 10 seconds`() = runTest {
        var calls = 0
        retryWithTimeout(timeoutPerAttempt = 5.seconds, policy = RetryPolicy(maxRetries = 3, initialBackoff = 1.seconds)) {
            if (++calls == 1) throw quota()
            "ok"
        }
        // quotaBackoffs[0] = 15 s ±20% → 12..18 s (HEAD: 10 s hint-by-word + 1 s = 11 s)
        assertTrue(currentTime in 12_000..18_000, "first quota pause must follow the ladder (12..18 s); was ${currentTime}ms")
    }

    @Test
    fun `quota and other failures have separate budgets`() = runTest {
        var calls = 0
        val result = retryWithTimeout(timeoutPerAttempt = 5.seconds, policy = RetryPolicy(maxRetries = 2, initialBackoff = 1.seconds)) {
            when (++calls) {
                1, 2 -> throw quota()
                3, 4 -> throw other()
                else -> "ok"
            }
        }
        // a shared budget (2) would be exhausted by the third failure
        assertEquals("ok", result)
        assertEquals(5, calls)
    }

    @Test
    fun `per-attempt timeout is retried even when the strategy rejects NETWORK_TIMEOUT`() = runTest {
        val rejectTimeouts = RetryStrategy { t, _ -> (t as? TransientException)?.reason != TransientReason.NETWORK_TIMEOUT }
        var calls = 0
        val result = retryWithTimeout(
            timeoutPerAttempt = 1.seconds,
            policy = RetryPolicy(strategy = rejectTimeouts, maxRetries = 2, initialBackoff = 1.seconds),
        ) {
            if (++calls == 1) delay(10.seconds)
            "ok"
        }
        assertEquals("ok", result)
        assertEquals(2, calls, "the timed-out attempt is retried: the timeout is decided by the wrapper, not by the strategy")
    }
}
