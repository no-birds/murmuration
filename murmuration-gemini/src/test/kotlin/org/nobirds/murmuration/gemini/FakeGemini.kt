package org.nobirds.murmuration.gemini

import java.io.InputStream
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import kotlin.concurrent.thread

/**
 * Raw HTTP server standing in for Gemini in tests that need a REAL genai `Client` round-trip
 * (a mock of the SDK would hide transport behaviour: timeouts, cancelled calls, silence).
 * Records requests, replies per [respond], and signals when the client closes the socket.
 *
 * Canonical samples: `StreamCallTrackerTest.ttfbTimeoutCancelsTheCall` (silent server),
 * `GeminiPromptStartedBeforeFirstChunkTest` (silent server through `GeminiClient`).
 */
internal class FakeGemini(private val respond: (Socket) -> Unit) : AutoCloseable {
    private val server = ServerSocket(0)
    val requests = CopyOnWriteArrayList<String>()
    val clientClosed = CountDownLatch(1)
    val port: Int get() = server.localPort

    init {
        thread(isDaemon = true) {
            while (!server.isClosed) {
                val socket = runCatching { server.accept() }.getOrNull() ?: break
                thread(isDaemon = true) { serve(socket) }
            }
        }
    }

    private fun serve(socket: Socket) {
        val input = socket.getInputStream()
        requests += readHeaders(input)
        respond(socket)
        // Blocks until the client hangs up (-1) or the connection resets — i.e. until Call.cancel() lands.
        runCatching { while (input.read() != -1) { /* drain */ } }
        clientClosed.countDown()
        runCatching { socket.close() }
    }

    private fun readHeaders(input: InputStream): String {
        val sb = StringBuilder()
        while (!sb.endsWith("\r\n\r\n")) {
            val b = input.read()
            if (b == -1) break
            sb.append(b.toChar())
        }
        // The POST body follows the headers; headers alone are enough for these tests.
        return sb.toString()
    }

    override fun close() = server.close()
}
