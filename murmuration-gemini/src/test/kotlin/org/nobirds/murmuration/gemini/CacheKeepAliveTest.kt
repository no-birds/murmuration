package org.nobirds.murmuration.gemini

import kotlinx.coroutines.delay
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptCachingProvider
import java.util.concurrent.atomic.AtomicInteger
import kotlin.time.Clock
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds
import kotlin.time.Instant

class CacheKeepAliveTest {

    private class CountingProvider(
        private val behavior: (callIndex: Int) -> Instant? = { Clock.System.now() + 10.minutes },
    ) : PromptCachingProvider {
        val keepAliveCount = AtomicInteger(0)

        override suspend fun processPrompt(prompt: Prompt): PromptCachingProvider.CacheResult =
            PromptCachingProvider.CacheResult(prompt = prompt)

        override suspend fun keepAlive(cacheId: String): Instant? {
            val idx = keepAliveCount.incrementAndGet()
            return behavior(idx)
        }
    }

    @Test
    fun `keeper polls provider periodically while block runs`() = runTest {
        val provider = CountingProvider()
        val interval = 60.seconds
        val workDuration = 5.minutes

        val result = provider.withCacheKeepAlive("cache-id", interval) {
            delay(workDuration)
            "ok"
        }

        assertEquals("ok", result)
        // After 5 minutes with a 60-second interval we expect 4-5 keepAlive polls (boundary).
        val count = provider.keepAliveCount.get()
        assertTrue(count in 4..5, "expected 4..5 keep-alive polls, got $count")
    }

    @Test
    fun `keeper stops polling once provider returns null`() = runTest {
        // On the third iteration the provider reports the cache expired (null) — the keeper must stop.
        val provider = CountingProvider(behavior = { idx ->
            if (idx >= 3) null else Clock.System.now() + 10.minutes
        })

        val result = provider.withCacheKeepAlive("cache-id", 60.seconds) {
            delay(10.minutes)
            "ok"
        }

        assertEquals("ok", result)
        // 3 calls — on the third we got null and stopped.
        assertEquals(3, provider.keepAliveCount.get())
    }

    @Test
    fun `keeper does not run when cacheId is null`() = runTest {
        val provider = CountingProvider()

        val result = provider.withCacheKeepAlive(null, 60.seconds) {
            delay(10.minutes)
            "ok"
        }

        assertEquals("ok", result)
        assertEquals(0, provider.keepAliveCount.get())
    }

    @Test
    fun `keeper does not run when provider is null`() = runTest {
        val result = null.withCacheKeepAlive("cache-id", 60.seconds) {
            delay(10.minutes)
            "ok"
        }
        assertEquals("ok", result)
    }

    @Test
    fun `keeper is cancelled when block throws`() = runTest {
        val provider = CountingProvider()
        val thrown = runCatching {
            provider.withCacheKeepAlive("cache-id", 60.seconds) {
                delay(2.minutes)
                error("boom")
            }
        }
        assertTrue(thrown.isFailure)
        // After cancellation the keeper must make no new calls; the counter stays <= 2.
        val count = provider.keepAliveCount.get()
        assertTrue(count <= 2, "expected ≤ 2 keep-alive polls after exception, got $count")
    }

    @Test
    fun `keeper survives transient keepAlive exception and keeps polling`() = runTest {
        var calls = 0
        val provider = object : PromptCachingProvider {
            override suspend fun processPrompt(prompt: Prompt): PromptCachingProvider.CacheResult =
                PromptCachingProvider.CacheResult(prompt = prompt)

            override suspend fun keepAlive(cacheId: String): Instant {
                calls++
                if (calls == 2) error("transient gemini error")
                return Clock.System.now() + 10.minutes
            }
        }

        val result = provider.withCacheKeepAlive("cache-id", 60.seconds) {
            delay(5.minutes)
            "ok"
        }

        assertEquals("ok", result)
        // Despite one failed iteration, the keeper keeps working and makes 4-5 calls.
        assertTrue(calls in 4..5, "expected keeper to keep going after transient failure, got $calls calls")
    }
}
