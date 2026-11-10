package org.nobirds.murmuration.prompt.cache

import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.PromptCacheScope
import org.nobirds.murmuration.prompt.PromptContextCacheEntry
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.time.Instant

class CacheGrowthStrategyTest {

    private fun entry(scope: PromptCacheScope) = PromptContextCacheEntry(
        cacheId = "cache-1",
        contentHash = "hash-1",
        model = "test-model",
        scope = scope,
        expirationTime = Instant.fromEpochMilliseconds(10_000),
        createTime = Instant.fromEpochMilliseconds(0),
        tokenCount = 1000,
    )

    private val epochGrow = CacheGrowthStrategy.EpochGrow(ttlBasedThresholdBytes = 64 * 1024)
    private val minBytes = 4 * 1024

    @Test
    fun `pipeline match keeps AlwaysGrow semantics`() {
        // Remainder above minCacheSizeBytes — grow, exactly like AlwaysGrow.
        assertTrue(epochGrow.shouldGrow(100_000, 100_000 + minBytes, minBytes, entry(PromptCacheScope.PIPELINE)))
        // Remainder below the minimum — no grow.
        assertFalse(epochGrow.shouldGrow(100_000, 100_000 + minBytes - 1, minBytes, entry(PromptCacheScope.PIPELINE)))
    }

    @Test
    fun `ttl-based match rebases only past the epoch threshold`() {
        val matched = 300_000
        // A typical single-run delta (a few KB of new history) must NOT trigger a re-create.
        assertFalse(epochGrow.shouldGrow(matched, matched + 10_000, minBytes, entry(PromptCacheScope.TTL_BASED)))
        // Accumulated remainder at the threshold — rebase (new epoch).
        assertTrue(epochGrow.shouldGrow(matched, matched + 64 * 1024, minBytes, entry(PromptCacheScope.TTL_BASED)))
    }

    @Test
    fun `always and never grow are unchanged`() {
        assertTrue(CacheGrowthStrategy.AlwaysGrow.shouldGrow(0, minBytes, minBytes, entry(PromptCacheScope.PIPELINE)))
        assertFalse(CacheGrowthStrategy.NeverGrow.shouldGrow(0, 1_000_000, minBytes, entry(PromptCacheScope.TTL_BASED)))
    }
}
