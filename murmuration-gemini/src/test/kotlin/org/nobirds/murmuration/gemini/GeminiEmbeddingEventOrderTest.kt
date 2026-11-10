package org.nobirds.murmuration.gemini

import com.google.genai.Client
import com.google.genai.types.HttpOptions
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutionEvent.UsageDelta
import org.nobirds.murmuration.prompt.EmbeddingPromptSettings
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import kotlin.test.assertTrue

/**
 * the embedding path emits `PromptStarted` -> `StreamResponded` -> `UsageDelta` ->
 * parts, like every other path. The fake answers in the Vertex wire format with `billableCharacterCount`,
 * which is what makes the executor emit a `UsageDelta` at all.
 *
 * RED on HEAD: `executeEmbedding` emits `UsageDelta` BEFORE `StreamResponded` (review F2).
 */
class GeminiEmbeddingEventOrderTest {

    private val servers = mutableListOf<FakeGemini>()

    @AfterEach
    fun tearDown() = servers.forEach { it.close() }

    @Test
    fun `embedding - StreamResponded comes right after PromptStarted, before usage`() {
        val body = """{"predictions":[{"embeddings":{"values":[0.1,0.2,0.3],"statistics":{"token_count":3,"truncated":false}}}],"metadata":{"billableCharacterCount":12}}"""
        val gemini = FakeGemini { socket ->
            socket.getOutputStream().apply {
                write(
                    ("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ${body.toByteArray().size}\r\n" +
                        "Connection: close\r\n\r\n$body").toByteArray(),
                )
                flush()
            }
            socket.shutdownOutput()
        }.also { servers += it }
        val sdk = Client.builder()
            .vertexAI(true)
            .apiKey("test-key")
            .httpOptions(HttpOptions.builder().baseUrl("http://127.0.0.1:${gemini.port}").build())
            .build()
        val prompt = buildPrompt {
            llm = GeminiLlms.Gemini.GEMINI_3_5_FLASH
            settings = EmbeddingPromptSettings()
            user { text("hello") }
        }

        val events = runBlocking { GeminiClient(sdk, GeminiWireKeyLayout.NONE).execute(prompt).toList() }

        assertTrue(events.any { it is UsageDelta }, "the fake must make the executor emit a UsageDelta (else the order is not exercised); events=${events.names()}")
        assertPromptStartedFirstAndOnce(events)
        assertStreamRespondedOnce(events)
        assertTrue(events.last() is PromptExecutionEvent.Completed, "events=${events.names()}")
    }
}
