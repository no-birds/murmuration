package org.nobirds.murmuration.retry

import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.QuotaExceededException
import org.nobirds.murmuration.prompt.QuotaKind
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

/** Budgets and pauses of [RetrySession] for quota errors (429 / RESOURCE_EXHAUSTED). */
class RetrySessionQuotaTest {

    private fun quota(retryAfter: Duration? = null) =
        QuotaExceededException(QuotaKind.RATE_LIMIT, retryAfter, "429 RESOURCE_EXHAUSTED", null)

    private fun timeout() = TransientException(TransientReason.NETWORK_TIMEOUT, "ttfb", null)

    private fun session(random: Random = Random(1)) = RetryPolicy().newSession(random)

    @Test
    fun `quota pauses grow 15 30 60 within jitter`() {
        val s = session()
        val bases = listOf(15, 30, 60)
        val delays = bases.map { assertNotNull(s.nextDelayOrNull(quota())) }
        bases.zip(delays).forEach { (base, d) ->
            assertTrue(d >= base.seconds * 0.8 && d <= base.seconds * 1.2, "base=$base delay=$d out of ±20%")
        }
        assertTrue(delays[0] < delays[1] && delays[1] < delays[2], "pauses must grow: $delays")
    }

    @Test
    fun `jitter spans the whole range`() {
        val low = RetryPolicy().newSession(FixedRandom(0.0)).nextDelayOrNull(quota())
        val high = RetryPolicy().newSession(FixedRandom(1.0)).nextDelayOrNull(quota())
        assertEquals(12.seconds, low)
        assertEquals(18.seconds, high)
    }

    @Test
    fun `server retryAfter is honoured over the growing ladder`() {
        val s = session()
        assertEquals(8.seconds + 1.seconds, s.nextDelayOrNull(quota(retryAfter = 8.seconds)))
        assertEquals(40.seconds + 1.seconds, s.nextDelayOrNull(quota(retryAfter = 40.seconds)))
    }

    @Test
    fun `server retryAfter is capped at 120 seconds without jitter`() {
        val s = session()
        assertEquals(120.seconds + 1.seconds, s.nextDelayOrNull(quota(retryAfter = 3600.seconds)))
    }

    @Test
    fun `quota and timeout budgets are separate`() {
        val s = session()
        repeat(2) { assertNotNull(s.nextDelayOrNull(quota())) }
        repeat(2) { assertNotNull(s.nextDelayOrNull(timeout())) }
        // the shared budget (3) would have been exhausted by four errors; separate ones are not
        assertNotNull(s.nextDelayOrNull(quota()))
        assertNotNull(s.nextDelayOrNull(timeout()))
    }

    @Test
    fun `fourth quota exhausts its own budget`() {
        val s = session()
        repeat(3) { assertNotNull(s.nextDelayOrNull(quota())) }
        assertNull(s.nextDelayOrNull(quota()))
    }

    private class FixedRandom(private val value: Double) : Random() {
        override fun nextBits(bitCount: Int): Int = 0
        override fun nextDouble(): Double = value
    }
}
