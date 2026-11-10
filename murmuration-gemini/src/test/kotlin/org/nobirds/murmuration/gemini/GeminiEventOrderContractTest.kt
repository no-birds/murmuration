package org.nobirds.murmuration.gemini

import com.google.genai.Client
import com.google.genai.types.HttpOptions
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import java.net.Socket
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds

/**
 * Gemini side: event-order contract of [GeminiClient] with both response strategies
 * (streaming and direct) — one `PromptStarted`, first, on success / TTFB timeout / mid-stream break;
 * `StreamResponded` once, between `PromptStarted` and the first part.
 *
 * Mutants this contract is meant to kill (checked by hand when the emitter changes):
 * `late_prompt_started` (covered by [GeminiPromptStartedBeforeFirstChunkTest]),
 * `second_prompt_started_on_chunk` (`PromptStarted` count = 1 here on every path),
 * `responded_before_started` (`assertStreamRespondedOnce` order check).
 */
class GeminiEventOrderContractTest {

    private val servers = mutableListOf<FakeGemini>()

    @AfterEach
    fun tearDown() = servers.forEach { it.close() }

    private fun fake(respond: (Socket) -> Unit): FakeGemini = FakeGemini(respond).also { servers += it }

    private fun clientFor(fake: FakeGemini, tracker: StreamCallTracker?): Client = Client.builder()
        .apiKey("test-key")
        .httpOptions(HttpOptions.builder().baseUrl("http://127.0.0.1:${fake.port}").build())
        .apply { tracker?.let { clientOptions(it.clientOptions()) } }
        .build()

    private val prompt: Prompt = buildPrompt {
        llm = GeminiLlms.Gemini.GEMINI_3_5_FLASH
        user { text("hi") }
    }

    /** Goes through [GeminiClient], not the bare strategy: `PromptStarted` is the executor's event. */
    private fun run(sdk: Client, strategy: GeminiResponseStrategy): List<PromptExecutionEvent> = runBlocking {
        val events = mutableListOf<PromptExecutionEvent>()
        try {
            GeminiClient(sdk, GeminiWireKeyLayout.NONE, responseStrategy = strategy).execute(prompt).toList(events)
        } catch (_: Throwable) {
            // The terminal failure is visible through the Failed event.
        }
        events
    }

    private fun runStreaming(gemini: FakeGemini, idle: Duration = 700.milliseconds): List<PromptExecutionEvent> {
        val tracker = StreamCallTracker()
        val sdk = clientFor(gemini, tracker)
        return run(sdk, StreamingResponseStrategy(sdk, tracker, idleTimeout = idle))
    }

    private fun runDirect(gemini: FakeGemini): List<PromptExecutionEvent> {
        val sdk = clientFor(gemini, tracker = null)
        return run(sdk, DirectResponseStrategy(sdk))
    }

    private fun sseHead(socket: Socket) = socket.getOutputStream().apply {
        write("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n".toByteArray())
        flush()
    }

    private fun sseChunk(socket: Socket, text: String, finish: Boolean) {
        val finishField = if (finish) ""","finishReason":"STOP"""" else ""
        socket.getOutputStream().apply {
            write(
                ("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"$text\"}]}$finishField}]," +
                    "\"modelVersion\":\"served-model\"}\n\n").toByteArray(),
            )
            flush()
        }
    }

    private val successfulStream: (Socket) -> Unit = { socket ->
        sseHead(socket)
        sseChunk(socket, "one", finish = false)
        sseChunk(socket, "two", finish = true)
        socket.shutdownOutput()
    }

    // --- streaming ---

    @Test
    fun `stream success - one PromptStarted first, parts follow`() {
        val events = runStreaming(fake(successfulStream), idle = 5.seconds)

        assertPromptStartedFirstAndOnce(events)
        assertTrue(events.any { it is PartReceived }, "expected parts; events=${events.names()}")
        assertTrue(events.last() is PromptExecutionEvent.Completed, "events=${events.names()}")
    }

    @Test
    fun `stream TTFB timeout - one PromptStarted before Failed`() {
        val events = runStreaming(fake { /* silent */ })

        assertPromptStartedFirstAndOnce(events)
        assertTrue(events.last() is PromptExecutionEvent.Failed, "events=${events.names()}")
    }

    @Test
    fun `stream break after the first chunk - still one PromptStarted`() {
        val events = runStreaming(
            fake { socket ->
                sseHead(socket)
                sseChunk(socket, "partial-1", finish = false)
                sseChunk(socket, "partial-2", finish = false)
                // then silence: inter-chunk idle timeout
            },
        )

        assertPromptStartedFirstAndOnce(events)
        assertEquals(1, events.count { it is PromptStarted })
        assertTrue(events.last() is PromptExecutionEvent.Failed, "events=${events.names()}")
    }

    @Test
    fun `stream success - StreamResponded once between PromptStarted and the first part`() {
        assertStreamRespondedOnce(runStreaming(fake(successfulStream), idle = 5.seconds))
    }

    // --- direct ---

    private fun jsonResponse(socket: Socket) {
        val body = """{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],"modelVersion":"served-model"}"""
        socket.getOutputStream().apply {
            write(
                ("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ${body.toByteArray().size}\r\n" +
                    "Connection: close\r\n\r\n$body").toByteArray(),
            )
            flush()
        }
        socket.shutdownOutput()
    }

    @Test
    fun `direct success - one PromptStarted first, parts follow`() {
        val events = runDirect(fake(::jsonResponse))

        assertPromptStartedFirstAndOnce(events)
        assertTrue(events.any { it is PartReceived }, "expected parts; events=${events.names()}")
        assertTrue(events.last() is PromptExecutionEvent.Completed, "events=${events.names()}")
    }

    @Test
    fun `direct success - StreamResponded once between PromptStarted and the first part`() {
        assertStreamRespondedOnce(runDirect(fake(::jsonResponse)))
    }
}
