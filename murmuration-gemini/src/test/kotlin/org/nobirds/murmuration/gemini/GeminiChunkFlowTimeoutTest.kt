package org.nobirds.murmuration.gemini

import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds

/**
 * Fix 2 (streaming path): the overall round-trip cap sits ON TOP of the per-chunk idle timeout.
 * A stream that keeps trickling chunks resets idle forever, so only the total-wall-time deadline
 * stops it. Exercised through [asTimedChunkFlow] with plain fake iterators — the genai
 * `ResponseStream`/`Client` cannot be faked (public-field chain, not mockable), so the timeout
 * loop was extracted to this iterator-level seam. Wall-clock based (`System.nanoTime()`), so these
 * use real (short) durations, not virtual time.
 */
class GeminiChunkFlowTimeoutTest {

    /** Infinite iterator that yields an incrementing Int every [perElementDelay]. */
    private class TricklingIterator(private val perElementDelay: Long) : Iterator<Int> {
        private var n = 0
        override fun hasNext(): Boolean {
            if (perElementDelay > 0) Thread.sleep(perElementDelay)
            return true
        }
        override fun next(): Int = n++
    }

    @Test
    fun `overall cap fires on an endlessly trickling stream`() = runBlocking {
        // idle is generous (never trips); each chunk arrives every 10ms, overall budget is 150ms.
        val ex = assertFailsWith<TransientException> {
            TricklingIterator(perElementDelay = 10)
                .asTimedChunkFlow(idleTimeout = 10.seconds, overallTimeout = 150.milliseconds, abortPull = {})
                .toList()
        }
        assertEquals(TransientReason.NETWORK_TIMEOUT, ex.reason)
        assertTrue(
            ex.message?.contains("overall round-trip budget") == true,
            "expected overall-budget message, got: ${ex.message}",
        )
    }

    @Test
    fun `happy path emits every chunk and completes when under budget`() = runBlocking {
        val chunks = listOf(10, 20, 30).iterator()
            .asTimedChunkFlow(idleTimeout = 5.seconds, overallTimeout = 5.seconds, abortPull = {})
            .toList()
        assertEquals(listOf(10, 20, 30), chunks)
    }

    @Test
    fun `overall budget binds the pull when idle exceeds the remaining budget`() = runBlocking {
        // idle (10s) is far larger than the overall budget (200ms). A chunk that stalls 10s would,
        // before the fix, be waited on for the full idle timeout — overshooting the deadline by ~10s.
        // Now the pull is capped at the remaining budget, so it trips near 200ms with an
        // overall-budget message, NOT an idle one.
        val startedAt = System.nanoTime()
        val ex = assertFailsWith<TransientException> {
            TricklingIterator(perElementDelay = 10_000)
                .asTimedChunkFlow(idleTimeout = 10.seconds, overallTimeout = 200.milliseconds, abortPull = {})
                .toList()
        }
        val elapsedMs = (System.nanoTime() - startedAt) / 1_000_000
        assertEquals(TransientReason.NETWORK_TIMEOUT, ex.reason)
        assertTrue(
            ex.message?.contains("overall round-trip budget") == true,
            "expected overall-budget message (not idle), got: ${ex.message}",
        )
        assertTrue(
            elapsedMs < 2_000,
            "should trip near the 200ms overall budget, not wait the 10s idle; waited ${elapsedMs}ms",
        )
    }

    @Test
    fun `idle timeout fires when a single chunk stalls`() = runBlocking {
        // hasNext blocks 800ms while idle budget is 100ms → idle trips before the chunk arrives.
        val ex = assertFailsWith<TransientException> {
            TricklingIterator(perElementDelay = 800)
                .asTimedChunkFlow(idleTimeout = 100.milliseconds, overallTimeout = 30.seconds, abortPull = {})
                .toList()
        }
        assertEquals(TransientReason.NETWORK_TIMEOUT, ex.reason)
        assertTrue(
            ex.message?.contains("idle") == true,
            "expected idle message, got: ${ex.message}",
        )
    }
}
