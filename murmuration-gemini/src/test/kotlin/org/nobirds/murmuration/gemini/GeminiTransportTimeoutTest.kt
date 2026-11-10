package org.nobirds.murmuration.gemini

import com.google.genai.types.GenerateContentConfig
import com.google.genai.types.HttpOptions
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import java.util.concurrent.CompletableFuture
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds

/**
 * Transport-level timeout guards on the Gemini path that don't need a live genai `Client`:
 *
 *  - Fix 1 (Direct callTimeout backstop): [GenerateContentConfig.withCallTimeout] must encode the
 *    OkHttp `callTimeout` as a per-request `HttpOptions.timeout` (ms) — the only handle the SDK
 *    exposes to actually abort a hung Direct call. The end-to-end socket abort itself needs a real
 *    server and was verified by reading google-genai 1.56.0 `HttpApiClient.asyncExecuteRequest`
 *    (it rebuilds the OkHttp client with `callTimeout` from the request's `HttpOptions` tag).
 *  - Fix 2 (streaming TTFB): [awaitFirstEventOrTimeout] must bound the first-event `await` and map a
 *    silent server to a retryable NETWORK_TIMEOUT.
 */
class GeminiTransportTimeoutTest {

    // --- Fix 1: withCallTimeout ---

    @Test
    fun `withCallTimeout encodes the timeout as HttpOptions timeout in millis`() {
        val config = GenerateContentConfig.builder().build()
        val withTimeout = config.withCallTimeout(630.seconds)

        val timeout = withTimeout.httpOptions().orElseThrow().timeout().orElseThrow()
        assertEquals(630_000, timeout)
    }

    @Test
    fun `withCallTimeout preserves existing HttpOptions fields`() {
        val config = GenerateContentConfig.builder()
            .httpOptions(HttpOptions.builder().apiVersion("v1beta1").build())
            .build()
        val withTimeout = config.withCallTimeout(600.seconds)

        val httpOptions = withTimeout.httpOptions().orElseThrow()
        assertEquals("v1beta1", httpOptions.apiVersion().orElseThrow())
        assertEquals(600_000, httpOptions.timeout().orElseThrow())
    }

    // --- Fix 2: awaitFirstEventOrTimeout (TTFB) ---

    @Test
    fun `TTFB await maps a never-resolving first event to a transient network timeout`() = runBlocking {
        val neverResolves = CompletableFuture<String>() // never completed → server accepted but silent
        val ex = assertFailsWith<TransientException> {
            awaitFirstEventOrTimeout(idleTimeout = 100.milliseconds, onTimeout = {}, future = neverResolves)
        }
        assertEquals(TransientReason.NETWORK_TIMEOUT, ex.reason)
        assertTrue(
            ex.message?.contains("TTFB") == true,
            "expected TTFB message, got: ${ex.message}",
        )
    }

    @Test
    fun `TTFB await returns the value when the first event resolves in time`() = runBlocking {
        val resolved = CompletableFuture.completedFuture("stream")
        val result = awaitFirstEventOrTimeout(idleTimeout = 5.seconds, onTimeout = {}, future = resolved)
        assertEquals("stream", result)
    }
}
