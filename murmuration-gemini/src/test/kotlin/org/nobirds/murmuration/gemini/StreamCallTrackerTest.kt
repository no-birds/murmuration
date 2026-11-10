package org.nobirds.murmuration.gemini

import com.google.genai.Client
import com.google.genai.types.Content
import com.google.genai.types.GenerateContentConfig
import com.google.genai.types.HttpOptions
import com.google.genai.types.Part
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import okhttp3.OkHttpClient
import okhttp3.Request
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import java.net.Socket
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds

/**
 * Zombie-Call guard of the streaming path: every stream timeout must cancel the
 * underlying OkHttp `Call` BEFORE the retryable failure propagates, so a retry never runs alongside
 * the timed-out request.
 *
 * The tests run a real genai [Client] (with [StreamCallTracker] wired in) against a raw local socket
 * server standing in for Gemini — the only way to prove the socket is REALLY torn down (a mock of the
 * Call would pass with a no-op `cancel`).
 *
 * Canonical sample for the tool: `first event never arrives -> aborted` ([ttfbTimeoutCancelsTheCall]).
 */
class StreamCallTrackerTest {

    private val servers = mutableListOf<FakeGemini>()

    @AfterEach
    fun tearDown() = servers.forEach { it.close() }

    private fun fake(respond: (Socket) -> Unit): FakeGemini = FakeGemini(respond).also { servers += it }

    private fun sseHead(socket: Socket) {
        socket.getOutputStream().apply {
            write("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n".toByteArray())
            flush()
        }
    }

    private fun sseChunk(socket: Socket, text: String, finish: Boolean) {
        val finishField = if (finish) ""","finishReason":"STOP"""" else ""
        socket.getOutputStream().apply {
            write(
                ("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"$text\"}]}$finishField}]}\n\n")
                    .toByteArray(),
            )
            flush()
        }
    }

    private fun clientFor(fake: FakeGemini, tracker: StreamCallTracker?): Client = Client.builder()
        .apiKey("test-key")
        .httpOptions(HttpOptions.builder().baseUrl("http://127.0.0.1:${fake.port}").build())
        .apply { tracker?.let { clientOptions(it.clientOptions()) } }
        .build()

    private val prompt: Prompt = buildPrompt {
        llm = GeminiLlms.Gemini.GEMINI_3_5_FLASH
        user { text("hi") }
    }

    private fun run(strategy: StreamingResponseStrategy): List<PromptExecutionEvent> = runBlocking {
        val events = mutableListOf<PromptExecutionEvent>()
        try {
            strategy.execute(
                prompt = prompt,
                modelName = "gemini-test",
                contents = listOf(Content.fromParts(Part.fromText("hi"))),
                config = GenerateContentConfig.builder().build(),
                cacheResult = null,
                requestMessages = emptyList(),
                requestedModelName = "gemini-test",
            ).toList(events)
        } catch (_: Throwable) {
            // The terminal failure is asserted through the Failed event the strategy emits first.
        }
        events
    }

    private fun List<PromptExecutionEvent>.failure(): PromptExecutionEvent.Failed =
        assertNotNull(filterIsInstance<PromptExecutionEvent.Failed>().singleOrNull(), "expected exactly one Failed event")

    // --- Canonical sample: TTFB timeout aborts the Call ---

    @Test
    fun ttfbTimeoutCancelsTheCall() {
        val tracker = StreamCallTracker()
        val gemini = fake { /* accepts the request, never answers: zero events */ }
        val strategy = StreamingResponseStrategy(clientFor(gemini, tracker), tracker, idleTimeout = 700.milliseconds)

        val failed = run(strategy).failure()

        val error = failed.error as TransientException
        assertEquals(TransientReason.NETWORK_TIMEOUT, error.reason)
        assertTrue(error.message!!.contains("TTFB"), error.message)
        // stream_diag: the timeout is recorded with 0 chunks received.
        assertEquals(0, assertNotNull(failed.streamDiagnostics).chunkCount)
        // The socket is really closed by the cancel — no zombie request left behind.
        assertTrue(gemini.clientClosed.await(5, TimeUnit.SECONDS), "server never saw the client hang up (zombie Call)")
        assertEquals(0, tracker.openCount)
    }

    @Test
    fun interChunkIdleTimeoutCancelsTheCall() {
        val tracker = StreamCallTracker()
        val gemini = fake { socket ->
            sseHead(socket)
            // The SDK iterator reads one event ahead, so two events are needed for the first to be
            // delivered; after them: silence.
            sseChunk(socket, "partial-1", finish = false)
            sseChunk(socket, "partial-2", finish = false)
        }
        val strategy = StreamingResponseStrategy(clientFor(gemini, tracker), tracker, idleTimeout = 700.milliseconds)

        val failed = run(strategy).failure()

        val error = failed.error as TransientException
        assertEquals(TransientReason.NETWORK_TIMEOUT, error.reason)
        assertTrue(error.message!!.contains("idle"), error.message)
        assertTrue(assertNotNull(failed.streamDiagnostics).chunkCount >= 1, "stream must have started before going idle")
        assertTrue(gemini.clientClosed.await(5, TimeUnit.SECONDS), "server never saw the client hang up (zombie Call)")
        assertEquals(0, tracker.openCount)
    }

    @Test
    fun healthyStreamCompletesAndTicketHeaderNeverReachesTheServer() {
        val tracker = StreamCallTracker()
        val gemini = fake { socket ->
            sseHead(socket)
            sseChunk(socket, "done", finish = true)
            socket.shutdownOutput()
        }
        val strategy = StreamingResponseStrategy(clientFor(gemini, tracker), tracker, idleTimeout = 5.seconds)

        val events = run(strategy)

        assertTrue(events.any { it is PromptExecutionEvent.Completed }, "stream should complete: $events")
        assertFalse(
            gemini.requests.any { it.contains(StreamCallTracker.TICKET_HEADER, ignoreCase = true) },
            "ticket header must be stripped before the request is sent",
        )
        assertEquals(0, tracker.openCount)
    }

    @Test
    fun clientWithoutTrackerFailsLoudly() {
        val tracker = StreamCallTracker()
        val gemini = fake { socket ->
            sseHead(socket)
            sseChunk(socket, "done", finish = true)
            socket.shutdownOutput()
        }
        // Client built WITHOUT tracker.clientOptions(): a timeout could not be aborted — must not run silently.
        val strategy = StreamingResponseStrategy(clientFor(gemini, tracker = null), tracker, idleTimeout = 5.seconds)

        val failed = run(strategy).failure()

        assertTrue(failed.error.message!!.contains("StreamCallTracker"), failed.error.message)
    }

    // --- Ordering: the Call is cancelled BEFORE the retryable exception propagates ---

    @Test
    fun onTimeoutRunsBeforeTheRetryableExceptionPropagates() = runBlocking {
        var cancelled = false
        val ex = runCatching {
            awaitFirstEventOrTimeout(
                idleTimeout = 100.milliseconds,
                onTimeout = { cancelled = true },
                future = CompletableFuture<String>(),
            )
        }.exceptionOrNull()

        assertTrue(ex is TransientException)
        assertTrue(cancelled, "abort must complete before the exception reaches the retry layer")
    }

    @Test
    fun abortBeforeRegistrationCancelsTheLateCall() {
        val tracker = StreamCallTracker()
        val ticket = tracker.open()
        ticket.abort() // raced the request start: nothing registered yet
        assertEquals(1, tracker.openCount, "ticket must survive until the late call registers")

        val late = OkHttpClient().newCall(Request.Builder().url("http://127.0.0.1:1/").build())
        ticket.register(late)

        assertTrue(late.isCanceled())
        assertEquals(0, tracker.openCount)
    }

    @Test
    fun applyToPreservesExistingHttpOptions() {
        val tracker = StreamCallTracker()
        val ticket = tracker.open()
        val base = GenerateContentConfig.builder()
            .httpOptions(HttpOptions.builder().apiVersion("v1").headers(mapOf("X-Other" to "1")).build())
            .build()

        val options = ticket.applyTo(base).httpOptions().orElseThrow()

        assertEquals("v1", options.apiVersion().orElseThrow())
        assertEquals("1", options.headers().orElseThrow()["X-Other"])
        assertEquals(ticket.id, options.headers().orElseThrow()[StreamCallTracker.TICKET_HEADER])
    }
}
