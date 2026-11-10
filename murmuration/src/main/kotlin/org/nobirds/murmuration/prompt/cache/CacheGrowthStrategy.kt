package org.nobirds.murmuration.prompt.cache

import org.nobirds.murmuration.prompt.PromptCacheScope
import org.nobirds.murmuration.prompt.PromptContextCacheEntry

/**
 * Manager decision: what to do when a prefix cache is found whose remainder (uncached
 * cacheable parts) is larger than the `minCacheSizeBytes` threshold.
 *
 * Default (current) behaviour is `AlwaysGrow`: each time we create a new long cache on top of
 * the found prefix. This gives the maximum cache_hit on subsequent round-trips of the same
 * agent, but costs a full-price write for every such "extension", and some of them may
 * never be read (for example, on the agent's last round-trip, after which
 * there will be no follow-up).
 *
 * Alternatives exist — but only after analysing real pipeline profiles.
 */
sealed interface CacheGrowthStrategy {
    /**
     * The decision is made by the provider. Parameters:
     * @param matchedSize total "weight" of the already cached parts (by the backend's estimateSize)
     * @param totalSize total "weight" of all cacheable parts of the current request
     * @param minCacheSizeBytes threshold for creating a new cache, from the manager config
     * @param matchedEntry the prefix cache found in storage
     */
    fun shouldGrow(
        matchedSize: Int,
        totalSize: Int,
        minCacheSizeBytes: Int,
        matchedEntry: PromptContextCacheEntry,
    ): Boolean

    /**
     * Current behaviour: if the remainder is above the threshold — grow. This is a safe default; any
     * experiments with "don't grow" / "grow only after reuse" are tried by explicitly plugging in
     * another strategy via `CacheManagerConfig.growthStrategy`.
     */
    object AlwaysGrow : CacheGrowthStrategy {
        override fun shouldGrow(
            matchedSize: Int,
            totalSize: Int,
            minCacheSizeBytes: Int,
            matchedEntry: PromptContextCacheEntry,
        ): Boolean = (totalSize - matchedSize) >= minCacheSizeBytes
    }

    /**
     * Never grow: reuse the found prefix as is, the remainder goes as a fresh tail
     * at full price. Experimental strategy — may make pipelines cheaper with many
     * one-off long caches (the agent's last round-trip etc.) at the cost of
     * a lower cache_hit on repeated round-trips. **Measure on dev before enabling.**
     */
    object NeverGrow : CacheGrowthStrategy {
        override fun shouldGrow(
            matchedSize: Int,
            totalSize: Int,
            minCacheSizeBytes: Int,
            matchedEntry: PromptContextCacheEntry,
        ): Boolean = false
    }

    /**
     * Scope-aware growth for long-lived caches ("epoch" rebasing).
     *
     * Provider caches cannot be extended in place — "growing" means creating a brand-new
     * cache and paying the write cost for the WHOLE content again. For short-lived
     * `PIPELINE` caches that trade-off is the [AlwaysGrow] one and stays unchanged here
     * (grow when the remainder exceeds `minCacheSizeBytes`).
     *
     * For `TTL_BASED` caches that are reused ACROSS pipeline runs (e.g. a per-game prefix
     * of book + rules + character codex + played-out history), small per-run deltas — one
     * new history entry per run — would otherwise trigger a full re-create on every run,
     * which is exactly the cost this scope tries to avoid. Instead the matched prefix is
     * reused and the delta rides as regular input until the accumulated remainder exceeds
     * [ttlBasedThresholdBytes]; only then a new longer cache is created (an "epoch").
     */
    data class EpochGrow(
        val ttlBasedThresholdBytes: Int = 64 * 1024,
    ) : CacheGrowthStrategy {
        override fun shouldGrow(
            matchedSize: Int,
            totalSize: Int,
            minCacheSizeBytes: Int,
            matchedEntry: PromptContextCacheEntry,
        ): Boolean {
            val remainder = totalSize - matchedSize
            return when (matchedEntry.scope) {
                PromptCacheScope.TTL_BASED -> remainder >= ttlBasedThresholdBytes
                else -> remainder >= minCacheSizeBytes
            }
        }
    }
}
