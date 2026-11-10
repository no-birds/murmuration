package org.nobirds.murmuration.gemini

import com.google.genai.Client
import com.google.genai.types.HttpOptions
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptCachingProvider
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Failed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds

/**
 * Regression guard: a Gemini round-trip whose server accepts the connection and never answers
 * (TTFB timeout, zero chunks) must STILL have announced what was sent: exactly one `PromptStarted`
 * carrying `cacheId`, `cacheStats` and the request messages, BEFORE the terminal `Failed`
 * (`chunkCount = 0`, `firstChunkMs = null`). Without it the dead round-trip is recorded in
 * `llm_statistics` with `cacheId = null` and `requestMessages = []`.
 *
 * Real [GeminiClient] + real genai [Client] + [StreamingResponseStrategy] against a silent raw
 * socket server ([FakeGemini]); only the [PromptCachingProvider] is faked (it yields a cache id).
 *
 * Failure mode guarded: `PromptStarted` built from the first chunk, which never arrives; it must be
 * emitted before the request is sent.
 */
class GeminiPromptStartedBeforeFirstChunkTest {

    private val servers = mutableListOf<FakeGemini>()

    @AfterEach
    fun tearDown() = servers.forEach { it.close() }

    private class FakeCachingProvider : PromptCachingProvider {
        override suspend fun processPrompt(prompt: Prompt) = PromptCachingProvider.CacheResult(
            id = CACHE_ID,
            prompt = prompt,
            statistics = PromptCachingProvider.CacheStatistics(tokenCount = CACHED_TOKENS),
        )
    }

    private val prompt: Prompt = buildPrompt {
        llm = GeminiLlms.Gemini.GEMINI_3_5_FLASH
        user { text("hi") }
    }

    private fun silentClient(): GeminiClient {
        val gemini = FakeGemini { /* accepts the request, never answers: zero chunks */ }.also { servers += it }
        val tracker = StreamCallTracker()
        val sdk = Client.builder()
            .apiKey("test-key")
            .httpOptions(HttpOptions.builder().baseUrl("http://127.0.0.1:${gemini.port}").build())
            .clientOptions(tracker.clientOptions())
            .build()
        return GeminiClient(
            geminiClient = sdk,
            wireKeyLayout = GeminiWireKeyLayout.NONE,
            cachingProvider = FakeCachingProvider(),
            responseStrategy = StreamingResponseStrategy(sdk, tracker, idleTimeout = 700.milliseconds),
        )
    }

    private fun collect(client: GeminiClient): List<PromptExecutionEvent> = runBlocking {
        val events = mutableListOf<PromptExecutionEvent>()
        try {
            client.execute(prompt).toList(events)
        } catch (_: Throwable) {
            // The terminal failure is asserted through the Failed event the executor emits first.
        }
        events
    }

    @Test
    fun `TTFB timeout before the first chunk still carries cacheId, cacheStats and request in PromptStarted`() {
        val events = collect(silentClient())

        val started = assertNotNull(
            events.filterIsInstance<PromptStarted>().singleOrNull(),
            "a round-trip that reached the wire must announce PromptStarted even with zero chunks; events=${events.names()}",
        )
        assertEquals(CACHE_ID, started.cacheId)
        assertEquals(CACHED_TOKENS, started.cacheStats?.tokenCount)
        assertTrue(started.requestMessages.isNotEmpty(), "requestMessages must be the request as sent, not empty")

        assertPromptStartedFirstAndOnce(events)
        val failed = events.filterIsInstance<Failed>().single()
        assertTrue(
            events.indexOf(started) < events.indexOf(failed),
            "PromptStarted must precede Failed; events=${events.names()}",
        )
        assertEquals(TransientReason.NETWORK_TIMEOUT, (failed.error as TransientException).reason)
        val diag = assertNotNull(failed.streamDiagnostics)
        assertEquals(0, diag.chunkCount)
        assertNull(diag.firstChunkMs)
    }

    @Test
    fun `TTFB timeout emits no StreamResponded - the provider never answered`() {
        // Name-based count: the event is not emitted before a chunk.
        assertEquals(0, collect(silentClient()).streamRespondedCount())
    }

    private companion object {
        const val CACHE_ID = "cachedContents/test-cache"
        const val CACHED_TOKENS = 4242L
    }
}
