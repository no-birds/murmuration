package org.nobirds.murmuration.gemini

import com.google.genai.Client
import com.google.genai.types.CreateCachedContentConfig
import com.google.genai.types.HttpOptions
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * the SDK's built-in `RetryInterceptor` (default 5 attempts on 408/429/5xx) must be off in the
 * production client, so our `RetrySession` is the only retry layer — one attempt of ours = one HTTP request.
 * The client is built exactly like production (`withoutSdkRetry()` on the client-level HttpOptions).
 */
class GenaiHttpOptionsTest {

    private val servers = mutableListOf<FakeGemini>()

    @AfterEach
    fun tearDown() = servers.forEach { it.close() }

    private fun tooManyRequests() = FakeGemini { socket ->
        val body = """{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"quota"}}"""
        socket.getOutputStream().apply {
            write(
                ("HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\n" +
                    "Content-Length: ${body.length}\r\nConnection: close\r\n\r\n$body").toByteArray(),
            )
            flush()
        }
    }.also { servers += it }

    private fun clientFor(fake: FakeGemini): Client = Client.builder()
        .apiKey("test-key")
        .httpOptions(HttpOptions.builder().baseUrl("http://127.0.0.1:${fake.port}").withoutSdkRetry().build())
        .build()

    @Test
    fun `retry options are attempts 1`() {
        val options = HttpOptions.builder().withoutSdkRetry().build()
        assertEquals(1, options.retryOptions().get().attempts().get())
    }

    @Test
    fun `generateContent on 429 makes exactly one HTTP request`() {
        val fake = tooManyRequests()
        assertFailsWith<Exception> { clientFor(fake).models.generateContent("gemini-2.5-flash", "hi", null) }
        assertEquals(1, fake.requests.size)
    }

    @Test
    fun `generateContentStream on 429 makes exactly one HTTP request`() {
        val fake = tooManyRequests()
        assertFailsWith<Exception> {
            clientFor(fake).models.generateContentStream("gemini-2.5-flash", "hi", null).use { it.forEach { } }
        }
        assertEquals(1, fake.requests.size)
    }

    @Test
    fun `caches create on 429 makes exactly one HTTP request`() {
        val fake = tooManyRequests()
        assertFailsWith<Exception> {
            clientFor(fake).caches.create("gemini-2.5-flash", CreateCachedContentConfig.builder().build())
        }
        assertEquals(1, fake.requests.size)
    }
}
