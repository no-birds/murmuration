package org.nobirds.murmuration.prompt.cache

import kotlin.time.Duration
import kotlin.time.Duration.Companion.minutes

/**
 * Cache manager config. Provider-specific parameters (for example, Gemini-only quotas)
 * do not belong here, but in the config of the specific backend.
 */
data class CacheManagerConfig(
    /** Base TTL for created caches. May be overridden by `PromptCacheStrategy.ttl`. */
    val ttl: Duration = 10.minutes,
    /** If `expirationTime - now < refreshDuration` — extend the cache before use. */
    val refreshDuration: Duration = 5.minutes,
    /**
     * Minimum size of the "increment" in bytes that justifies creating a new long cache
     * on top of a found prefix. Used by `CacheGrowthStrategy.AlwaysGrow`.
     */
    val minCacheSizeBytes: Int = 4 * 1024,
    /**
     * Which parts go into the cache: a strict prefix or a filter through NO_CACHE "holes".
     * Default `GroupByCacheability` collects all cacheable parts even across NO_CACHE "holes",
     * maximizing cache hit on pipelines with mixed strategies in one USER message.
     */
    val reorderStrategy: CacheReorderStrategy = GroupByCacheability,
    /** Whether to grow a long cache on top of a found prefix. */
    val growthStrategy: CacheGrowthStrategy = CacheGrowthStrategy.AlwaysGrow,
)
