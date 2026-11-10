package org.nobirds.murmuration.gemini

import com.google.genai.Client
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptCacheScope
import org.nobirds.murmuration.prompt.PromptCacheStrategy
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptContextCacheEntry
import org.nobirds.murmuration.prompt.PromptContextCacheStorage
import org.nobirds.murmuration.prompt.cache.GroupByCacheability
import org.nobirds.murmuration.prompt.cache.Strict
import org.nobirds.murmuration.prompt.TextPromptPart
import kotlin.time.Clock
import kotlin.time.Duration.Companion.hours
import kotlin.time.Duration.Companion.minutes

class GeminiPromptCachingProviderTest {

    private val client = mockk<Client>()
    private val storage = mockk<PromptContextCacheStorage>()
    private val provider = GeminiPromptCachingProvider(client, GeminiWireKeyLayout.NONE, storage)
    private val llm = Llm("google", "gemini", "gemini-3-flash", "v1", true, "gemini-3-flash-preview")

    @Test
    fun `test exact match`() = runBlocking {
        val content = listOf(
            PromptContent(PromptContentType.SYSTEM, listOf(TextPromptPart("Base", cacheStrategy = PromptCacheStrategy.PIPELINE)))
        )
        val prompt = Prompt(llm, null, content)

        // role:SYSTEMtext:Base -> role:SYSTEMtext:Base
        val hash = "bbb3afdb7f5e886f26023022d1181a651c8e25a7f80b439351ffaf74850dde2c"

        coEvery { storage.findByHash(any(), any()) } returns null
        coEvery { storage.findByHash(hash, any()) } returns PromptContextCacheEntry(
            cacheId = "cache-1",
            contentHash = hash,
            model = llm.model,
            scope = PromptCacheScope.PIPELINE,
            expirationTime = Clock.System.now() + 1.hours,
            createTime = Clock.System.now(),
            tokenCount = 100
        )

        val result = provider.processPrompt(prompt)

        assertEquals("cache-1", result.id)
        assertFalse(result.statistics?.isCreated ?: true)
        assertEquals(0, result.prompt.content.size)
    }

    @Test
    fun `test prefix match`() = runBlocking {
        val basePart = TextPromptPart("Base", cacheStrategy = PromptCacheStrategy.PIPELINE)
        val turnPart = TextPromptPart("Turn 1", cacheStrategy = PromptCacheStrategy.PIPELINE)

        val content = listOf(
            PromptContent(PromptContentType.SYSTEM, listOf(basePart)),
            PromptContent(PromptContentType.USER, listOf(turnPart))
        )
        val prompt = Prompt(llm, null, content)

        val baseHash = "bbb3afdb7f5e886f26023022d1181a651c8e25a7f80b439351ffaf74850dde2c"
        val fullHash = "07875b549e3a42da77fc83f7fb9ad97296451fa10345ecc00bdbb97786457753"

        coEvery { storage.findByHash(any(), any()) } returns null

        // Return cache for Base only
        coEvery { storage.findByHash(baseHash, any()) } returns PromptContextCacheEntry(
            cacheId = "cache-base",
            contentHash = baseHash,
            model = llm.model,
            scope = PromptCacheScope.PIPELINE,
            expirationTime = Clock.System.now() + 1.hours,
            createTime = Clock.System.now(),
            tokenCount = 50
        )

        val result = provider.processPrompt(prompt)

        assertEquals("cache-base", result.id)
        assertFalse(result.statistics?.isCreated ?: true)

        // Only Base part should be dropped
        assertEquals(1, result.prompt.content.size)
        assertEquals(PromptContentType.USER, result.prompt.content[0].type)
        assertEquals("Turn 1", (result.prompt.content[0].parts[0] as TextPromptPart).text)
    }

    @Test
    fun `group_by_cacheability collects pipeline parts through no_cache gap`() = runBlocking {
        val groupByProvider = GeminiPromptCachingProvider(
            client,
            GeminiWireKeyLayout.NONE,
            storage,
            config = GeminiCacheConfig(reorderStrategy = GroupByCacheability)
        )

        val pipe1 = TextPromptPart("PIPE_1", cacheStrategy = PromptCacheStrategy.PIPELINE)
        val gap = TextPromptPart("GAP_NC", cacheStrategy = PromptCacheStrategy.NO_CACHE)
        val pipe2 = TextPromptPart("PIPE_2", cacheStrategy = PromptCacheStrategy.PIPELINE)

        val content = listOf(
            PromptContent(PromptContentType.USER, listOf(pipe1, gap, pipe2))
        )
        val prompt = Prompt(llm, null, content)

        // Any hash → we land in a ready cache, to avoid hitting the Gemini API.
        coEvery { storage.findByHash(any(), any()) } returns PromptContextCacheEntry(
            cacheId = "cache-mixed",
            contentHash = "any",
            model = llm.model,
            scope = PromptCacheScope.PIPELINE,
            expirationTime = Clock.System.now() + 1.hours,
            createTime = Clock.System.now(),
            tokenCount = 100
        )

        val result = groupByProvider.processPrompt(prompt)

        assertEquals("cache-mixed", result.id)
        // PIPE_1 and PIPE_2 went into the cache through the GAP hole; only the NO_CACHE piece remains in fresh.
        assertEquals(1, result.prompt.content.size)
        val parts = result.prompt.content[0].parts
        assertEquals(1, parts.size)
        assertEquals("GAP_NC", (parts[0] as TextPromptPart).text)
    }

    @Test
    fun `strict reorder breaks on first no_cache and keeps trailing pipeline as fresh`() = runBlocking {
        val strictProvider = GeminiPromptCachingProvider(
            client,
            GeminiWireKeyLayout.NONE,
            storage,
            config = GeminiCacheConfig(reorderStrategy = Strict)
        )

        val pipe1 = TextPromptPart("PIPE_1", cacheStrategy = PromptCacheStrategy.PIPELINE)
        val gap = TextPromptPart("GAP_NC", cacheStrategy = PromptCacheStrategy.NO_CACHE)
        val pipe2 = TextPromptPart("PIPE_2", cacheStrategy = PromptCacheStrategy.PIPELINE)

        val content = listOf(
            PromptContent(PromptContentType.USER, listOf(pipe1, gap, pipe2))
        )
        val prompt = Prompt(llm, null, content)

        coEvery { storage.findByHash(any(), any()) } returns PromptContextCacheEntry(
            cacheId = "cache-strict",
            contentHash = "any",
            model = llm.model,
            scope = PromptCacheScope.PIPELINE,
            expirationTime = Clock.System.now() + 1.hours,
            createTime = Clock.System.now(),
            tokenCount = 50
        )

        val result = strictProvider.processPrompt(prompt)

        assertEquals("cache-strict", result.id)
        // STRICT breaks the chain at GAP — PIPE_2 does NOT make it into the cache, stays in fresh.
        // The order of fresh parts matches the original.
        assertEquals(1, result.prompt.content.size)
        val parts = result.prompt.content[0].parts
        assertEquals(2, parts.size)
        assertEquals("GAP_NC", (parts[0] as TextPromptPart).text)
        assertEquals("PIPE_2", (parts[1] as TextPromptPart).text)
    }

    @Test
    fun `group_by preserves relative order within cached and fresh halves across contents`() = runBlocking {
        val groupByProvider = GeminiPromptCachingProvider(
            client,
            GeminiWireKeyLayout.NONE,
            storage,
            config = GeminiCacheConfig(reorderStrategy = GroupByCacheability)
        )

        val sysPipe = TextPromptPart("SYS_PIPE", cacheStrategy = PromptCacheStrategy.PIPELINE)
        val userPipe1 = TextPromptPart("USER_PIPE_1", cacheStrategy = PromptCacheStrategy.PIPELINE)
        val userNc1 = TextPromptPart("USER_NC_1", cacheStrategy = PromptCacheStrategy.NO_CACHE)
        val userPipe2 = TextPromptPart("USER_PIPE_2", cacheStrategy = PromptCacheStrategy.PIPELINE)
        val userNc2 = TextPromptPart("USER_NC_2", cacheStrategy = PromptCacheStrategy.NO_CACHE)

        val content = listOf(
            PromptContent(PromptContentType.SYSTEM, listOf(sysPipe)),
            PromptContent(PromptContentType.USER, listOf(userPipe1, userNc1, userPipe2, userNc2))
        )
        val prompt = Prompt(llm, null, content)

        coEvery { storage.findByHash(any(), any()) } returns PromptContextCacheEntry(
            cacheId = "cache-order",
            contentHash = "any",
            model = llm.model,
            scope = PromptCacheScope.PIPELINE,
            expirationTime = Clock.System.now() + 1.hours,
            createTime = Clock.System.now(),
            tokenCount = 100
        )

        val result = groupByProvider.processPrompt(prompt)

        assertEquals("cache-order", result.id)
        // Fresh contains ONLY the NO_CACHE parts of the USER block, in the original relative order.
        assertEquals(1, result.prompt.content.size)
        assertEquals(PromptContentType.USER, result.prompt.content[0].type)
        val parts = result.prompt.content[0].parts
        assertEquals(2, parts.size)
        assertEquals("USER_NC_1", (parts[0] as TextPromptPart).text)
        assertEquals("USER_NC_2", (parts[1] as TextPromptPart).text)
    }

    @Test
    fun `keepAlive returns null when cache is missing`() = runBlocking {
        coEvery { storage.findById("missing") } returns null

        val result = provider.keepAlive("missing")

        assertNull(result, "missing cache must yield null to stop the underwriter")
    }

    @Test
    fun `keepAlive returns null when cache already expired`() = runBlocking {
        coEvery { storage.findById("dead") } returns PromptContextCacheEntry(
            cacheId = "dead",
            contentHash = "h",
            model = llm.model,
            scope = PromptCacheScope.PIPELINE,
            expirationTime = Clock.System.now() - 1.minutes,
            createTime = Clock.System.now() - 11.minutes,
            tokenCount = 100
        )

        val result = provider.keepAlive("dead")

        assertNull(result, "expired cache must yield null")
        // No trips to the extension API for a stale cache.
        coVerify(exactly = 0) { storage.updateExpiration(any(), any()) }
    }

    @Test
    fun `keepAlive does not call gemini when remaining is well above threshold`() = runBlocking {
        val expiration = Clock.System.now() + 9.minutes
        coEvery { storage.findById("fresh") } returns PromptContextCacheEntry(
            cacheId = "fresh",
            contentHash = "h",
            model = llm.model,
            scope = PromptCacheScope.PIPELINE,
            expirationTime = expiration,
            createTime = Clock.System.now() - 1.minutes,
            tokenCount = 100
        )

        val result = provider.keepAlive("fresh")

        // 9 minutes of headroom with threshold = 5 minutes — too early to extend, return as is.
        assertNotNull(result)
        assertEquals(expiration, result)
        coVerify(exactly = 0) { storage.updateExpiration(any(), any()) }
    }
}
