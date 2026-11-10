package org.nobirds.murmuration.gemini

import com.google.genai.Client
import com.google.genai.types.HttpOptions
import com.google.genai.types.HttpRetryOptions
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.PromptCacheScope
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.QuotaExceededException
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.retry.RetryPolicy
import java.net.Socket
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds
import kotlin.time.measureTime

/**
 * A raw genai 429 on cache create/refresh must take the quota ladder (not the "other" backoff):
 * the real [Client] talks to a fake server answering 429, the policy has a 300 ms quota ladder and a
 * 1 ms "other" backoff, so the elapsed time tells which class the failure landed in.
 */
class GeminiCacheBackendQuotaTest {

    private val policy = RetryPolicy(
        maxRetries = 2,
        initialBackoff = 1.milliseconds,
        maxBackoff = 1.milliseconds,
        quotaBackoffs = listOf(300.milliseconds),
        quotaJitter = 0.0,
    )

    private fun reply429(socket: Socket) {
        val body = """{"error":{"code":429,"message":"Resource exhausted. Quota exceeded","status":"RESOURCE_EXHAUSTED"}}"""
        socket.getOutputStream().apply {
            write("HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: ${body.length}\r\nConnection: close\r\n\r\n$body".toByteArray())
            flush()
        }
        socket.shutdownOutput()
    }

    private fun backendFor(fake: FakeGemini) = GeminiCacheBackend(
        Client.builder()
            .apiKey("test-key")
            .httpOptions(
                HttpOptions.builder()
                    .baseUrl("http://127.0.0.1:${fake.port}")
                    // The SDK retries 429 on its own; one wire attempt per backend attempt keeps the count exact.
                    .retryOptions(HttpRetryOptions.builder().attempts(1).build())
                    .build(),
            )
            .build(),
        GeminiWireKeyLayout.NONE,
        policy,
    )

    @Test
    fun `cache create on a raw 429 pauses by the quota ladder and ends with QuotaExceededException`() {
        FakeGemini(::reply429).use { fake ->
            val backend = backendFor(fake)
            val elapsed = measureTime {
                assertFailsWith<QuotaExceededException> {
                    runBlocking {
                        backend.createCache(
                            "gemini-test", listOf(PromptContent(PromptContentType.USER, listOf(TextPromptPart("x")))),
                            null, "hash", 5.minutes, PromptCacheScope.PIPELINE,
                        )
                    }
                }
            }
            assertEquals(3, fake.requests.size, "initial attempt + 2 quota retries")
            assertTrue(elapsed >= 550.milliseconds, "two 300 ms quota pauses expected, got $elapsed")
            assertTrue(elapsed < 20.seconds, "got $elapsed")
        }
    }

    @Test
    fun `cache refresh on a raw 429 pauses by the quota ladder and ends with QuotaExceededException`() {
        FakeGemini(::reply429).use { fake ->
            val backend = backendFor(fake)
            val elapsed = measureTime {
                assertFailsWith<QuotaExceededException> { runBlocking { backend.refreshCache("cachedContents/x", 5.minutes) } }
            }
            assertEquals(3, fake.requests.size)
            assertTrue(elapsed >= 550.milliseconds, "two 300 ms quota pauses expected, got $elapsed")
        }
    }
}
