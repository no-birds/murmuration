package org.nobirds.murmuration.prompt

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * `firstChunkMs`/`maxGapMs` used to be computed
 * only inside `if (logger.isDebugEnabled)`, for a log line — with DEBUG off the values did not
 * exist at all. [StreamTimingTracker] is the extracted, provider-neutral piece that now computes
 * them unconditionally; both `murmuration-gemini` and `murmuration-openai` feed it one
 * `recordChunk` per chunk arrival.
 *
 * The real Gemini/OpenAI SDK stream types are not fakeable (see `GeminiChunkFlowTimeoutTest`'s
 * comment on `genai Client`/`ResponseStream`), so this tracker is deliberately a pure, timestamp-
 * driven state machine — a "fake stream with an artificial pause" is exactly a hand-picked
 * sequence of `recordChunk` timestamps, no real sleeping or fake SDK plumbing required.
 */
class StreamTimingTrackerTest {

    @Test
    fun `before any chunk both values are null`() {
        val tracker = StreamTimingTracker(streamStartedAt = 1_000_000L)
        assertNull(tracker.firstChunkMs, "no chunk arrived yet")
        assertNull(tracker.maxGapMs, "need at least two chunks for a gap")
    }

    @Test
    fun `first chunk sets firstChunkMs but not maxGapMs`() {
        val streamStartedAt = 1_000_000L
        val tracker = StreamTimingTracker(streamStartedAt)
        // 30ms TTFB.
        tracker.recordChunk(streamStartedAt + 30_000_000L)
        assertEquals(30L, tracker.firstChunkMs)
        assertNull(tracker.maxGapMs, "still only one chunk — nothing to take a gap between")
    }

    @Test
    fun `an artificial pause between two chunks becomes maxGapMs`() {
        val streamStartedAt = 1_000_000L
        val tracker = StreamTimingTracker(streamStartedAt)
        tracker.recordChunk(streamStartedAt + 10_000_000L)   // chunk #1 at +10ms
        tracker.recordChunk(streamStartedAt + 510_000_000L)  // chunk #2 at +510ms — a 500ms pause
        assertEquals(10L, tracker.firstChunkMs, "firstChunkMs stays pinned to chunk #1")
        assertEquals(500L, tracker.maxGapMs, "the artificial 500ms pause is the max gap")
    }

    @Test
    fun `maxGapMs tracks the LARGEST gap, not the last one`() {
        val streamStartedAt = 1_000_000L
        val tracker = StreamTimingTracker(streamStartedAt)
        tracker.recordChunk(streamStartedAt + 10_000_000L)   // +10ms
        tracker.recordChunk(streamStartedAt + 310_000_000L)  // +310ms — gap #1: 300ms
        tracker.recordChunk(streamStartedAt + 330_000_000L)  // +330ms — gap #2: 20ms (smaller)
        assertEquals(300L, tracker.maxGapMs, "a smaller trailing gap must not shrink the max")
    }

    @Test
    fun `a zero gap between fast consecutive chunks is a measured zero, not null`() {
        val streamStartedAt = 1_000_000L
        val tracker = StreamTimingTracker(streamStartedAt)
        tracker.recordChunk(streamStartedAt + 10_000_000L)
        tracker.recordChunk(streamStartedAt + 10_000_000L) // same instant — legitimate 0ms gap
        assertEquals(0L, tracker.maxGapMs, "0ms is a real measurement, must not collapse to null")
    }

    /**
     * [StreamTimingTracker.snapshot] is the ONE
     * construction of [StreamDiagnostics] that used to be an identical private `buildStreamDiag`
     * copy-pasted into `murmuration-gemini` and `murmuration-openai`. The point being locked down
     * here is that it no longer needs `streamStartedAt`/`lastChunkAt` passed in from the caller —
     * the tracker already holds both (constructor arg, last [recordChunk] timestamp).
     */
    @Test
    fun `snapshot assembles StreamDiagnostics from counters passed in and timing tracked internally`() {
        val streamStartedAt = 1_000_000L
        val tracker = StreamTimingTracker(streamStartedAt)
        tracker.recordChunk(streamStartedAt + 10_000_000L)  // chunk #1 at +10ms
        tracker.recordChunk(streamStartedAt + 30_000_000L)  // chunk #2 at +30ms — gap 20ms

        val diag = tracker.snapshot(
            chunkCount = 2,
            totalTextParts = 3,
            totalThoughtParts = 1,
            totalToolCallParts = 4,
            lastFinishReason = "STOP",
        )

        assertEquals(2, diag.chunkCount)
        assertEquals(3, diag.partsText)
        assertEquals(1, diag.partsThought)
        assertEquals(4, diag.partsToolCall)
        assertEquals("STOP", diag.lastChunkFinishReason)
        assertEquals(10L, diag.firstChunkMs, "firstChunkMs comes from the tracker itself, not a param")
        assertEquals(20L, diag.maxGapMs, "maxGapMs comes from the tracker itself, not a param")
        // totalElapsedMs/sinceLastChunkMs are measured against `System.nanoTime()` at call time —
        // both are non-negative and totalElapsed is at least as large as sinceLastChunk, the one
        // structural invariant that does not depend on wall-clock timing of the test run itself.
        assertTrue(diag.totalElapsedMs >= 0L)
        assertTrue(diag.sinceLastChunkMs >= 0L)
        assertTrue(diag.totalElapsedMs >= diag.sinceLastChunkMs)
    }
}
