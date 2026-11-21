package org.nobirds.murmuration.prompt.cache

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.llm.llm
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptCacheScope
import org.nobirds.murmuration.prompt.PromptCacheStrategy
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptContextCacheEntry
import org.nobirds.murmuration.prompt.PromptContextCacheStorage
import org.nobirds.murmuration.prompt.PromptPart
import org.nobirds.murmuration.prompt.PromptSettings
import org.nobirds.murmuration.prompt.TextPromptPart
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.time.Duration
import kotlin.time.Instant

/**
 * [CacheManager] must not call `cache.create` for a model that declares implicit caching only
 * (`Caching(explicit = false)`), and must keep creating caches for a model that supports it.
 */
class CacheManagerExplicitCachingTest {

    private class CountingBackend : CacheBackend {
        var createCalls = 0

        override suspend fun createCache(
            model: String,
            contents: List<PromptContent>,
            settings: PromptSettings?,
            hash: String,
            ttl: Duration,
            scope: PromptCacheScope,
        ): CreateCacheResult {
            createCalls++
            return CreateCacheResult("caches/$createCalls", 1000, Instant.fromEpochSeconds(4_000_000_000))
        }

        override suspend fun refreshCache(cacheId: String, ttl: Duration): Instant = Instant.fromEpochSeconds(4_000_000_000)
        override suspend fun deleteCache(cacheId: String) = Unit
        override fun hashContent(contents: List<PromptContent>, settings: PromptSettings?): String =
            contents.hashCode().toString()
        override fun estimateSize(part: PromptPart): Int = (part as TextPromptPart).text.length
    }

    private class InMemoryStorage : PromptContextCacheStorage {
        val entries = mutableMapOf<String, PromptContextCacheEntry>()
        override suspend fun findByHash(hash: String, model: String) = entries.values.firstOrNull { it.contentHash == hash && it.model == model }
        override suspend fun findById(cacheId: String) = entries[cacheId]
        override suspend fun save(entry: PromptContextCacheEntry) { entries[entry.cacheId] = entry }
        override suspend fun updateExpiration(cacheId: String, expirationTime: Instant) = Unit
        override suspend fun delete(cacheId: String) { entries.remove(cacheId) }
    }

    private fun model(explicit: Boolean): Llm = llm("test", if (explicit) "explicit-model" else "implicit-model") {
        family = "test"; name = "test"; version = "1"
        input { text() }
        output { text() }
        caching(minTokens = 1, explicit = explicit)
    }

    private fun bigPrompt(llm: Llm) = Prompt(
        llm,
        settings = null,
        content = listOf(
            PromptContent(
                PromptContentType.SYSTEM,
                listOf(TextPromptPart("x".repeat(10_000), cacheStrategy = PromptCacheStrategy.PIPELINE)),
            ),
        ),
    )

    @Test
    fun `implicit-only model never reaches cache create and no cache is stored`() = runBlocking {
        val backend = CountingBackend()
        val storage = InMemoryStorage()
        val prompt = bigPrompt(model(explicit = false))

        val result = CacheManager(backend, storage).processPrompt(prompt)

        assertEquals(0, backend.createCalls)
        assertEquals(prompt, result.prompt)
        assertNull(result.id)
        assertNull(result.statistics)
        assertEquals(0, storage.entries.size)
    }

    @Test
    fun `explicit-capable model still creates a cache`() = runBlocking {
        val backend = CountingBackend()
        val storage = InMemoryStorage()

        val result = CacheManager(backend, storage).processPrompt(bigPrompt(model(explicit = true)))

        assertEquals(1, backend.createCalls)
        assertEquals(1, storage.entries.size)
        assertEquals("caches/1", result.id)
        assertEquals(true, result.statistics?.isCreated)
    }
}
