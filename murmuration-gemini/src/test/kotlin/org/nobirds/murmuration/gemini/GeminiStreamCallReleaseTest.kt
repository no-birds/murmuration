package org.nobirds.murmuration.gemini

import com.google.genai.Client
import com.google.genai.types.HttpOptions
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import kotlin.test.assertEquals
import kotlin.time.Duration.Companion.seconds

/**
 * Regression guard: a collector that goes away exactly while `PromptStarted` is delivered (pipeline
 * cancelled, agent abort) must not leave the stream's ticket in [StreamCallTracker] forever.
 *
 * Failure mode guarded: `callTracker.open()` and `emit(startedEvent)` sitting BEFORE the `try/finally`
 * that aborts the call, so the cancellation at `emit` skips the release (`open()` + `emit` belong inside `try`).
 */
class GeminiStreamCallReleaseTest {

    private val servers = mutableListOf<FakeGemini>()

    @AfterEach
    fun tearDown() = servers.forEach { it.close() }

    @Test
    fun `collector cancelled on PromptStarted leaves no open ticket`() {
        val gemini = FakeGemini { /* silent: the request never gets an answer */ }.also { servers += it }
        val tracker = StreamCallTracker()
        val sdk = Client.builder()
            .apiKey("test-key")
            .httpOptions(HttpOptions.builder().baseUrl("http://127.0.0.1:${gemini.port}").build())
            .clientOptions(tracker.clientOptions())
            .build()
        val prompt = buildPrompt {
            llm = GeminiLlms.Gemini.GEMINI_3_5_FLASH
            user { text("hi") }
        }
        val client = GeminiClient(sdk, GeminiWireKeyLayout.NONE, responseStrategy = StreamingResponseStrategy(sdk, tracker, idleTimeout = 5.seconds))

        runBlocking { client.execute(prompt).first { it is PromptStarted } }

        assertEquals(0, tracker.openCount, "the ticket opened for this stream must be released when the collector leaves at PromptStarted")
    }
}
