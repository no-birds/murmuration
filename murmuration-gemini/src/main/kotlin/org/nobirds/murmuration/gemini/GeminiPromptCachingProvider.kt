package org.nobirds.murmuration.gemini

import com.google.genai.Client
import org.nobirds.murmuration.prompt.PromptCachingProvider
import org.nobirds.murmuration.prompt.PromptContextCacheStorage
import org.nobirds.murmuration.prompt.cache.CacheGrowthStrategy
import org.nobirds.murmuration.prompt.cache.CacheManager
import org.nobirds.murmuration.prompt.cache.CacheManagerConfig
import org.nobirds.murmuration.prompt.cache.CacheReorderStrategy
import org.nobirds.murmuration.prompt.cache.GroupByCacheability
import kotlin.time.Duration
import kotlin.time.Duration.Companion.minutes

/**
 * Thin wrapper: binds the shared `CacheManager` to the Gemini-specific `GeminiCacheBackend`.
 * The management logic itself (what to put in the cache, how to look up, when to extend) lives in `CacheManager`,
 * I/O and serialization of parts into the Gemini format live in `GeminiCacheBackend`.
 *
 * Delegates all `PromptCachingProvider` methods via `by` delegation, so it implements nothing
 * extra and adds no behavior of its own.
 */
class GeminiPromptCachingProvider(
    client: Client,
    wireKeyLayout: GeminiWireKeyLayout,
    storage: PromptContextCacheStorage,
    config: GeminiCacheConfig = GeminiCacheConfig(),
) : PromptCachingProvider by CacheManager(
    backend = GeminiCacheBackend(client, wireKeyLayout),
    storage = storage,
    config = config.toCacheManagerConfig(),
)

/**
 * Config published by the Gemini build. Essentially a thin layer over `CacheManagerConfig`,
 * kept for backward compatibility of the name. If the Gemini backend later gets
 * provider-specific parameters, add them here; shared cacher settings live in
 * `CacheManagerConfig`.
 */
data class GeminiCacheConfig(
    val ttl: Duration = 10.minutes,
    val refreshDuration: Duration = 5.minutes,
    val minCacheSizeBytes: Int = 4 * 1024,
    /**
     * Default `GroupByCacheability` (not `Strict`), because in our pipelines
     * cacheable schemas/instructions often end up after NO_CACHE blocks (tasks,
     * snapshots) within a single USER message. Strict stops at the first NO_CACHE and does not
     * pick up a schema lying further on — which is tens of thousands of uncached tokens
     * (the `jsonWithSchema` schema part ~9K tokens × N round-trips). With GroupByCacheability
     * all PIPELINE parts move into the cache prefix across NO_CACHE "holes", NO_CACHE is the fresh tail.
     * The order of parts as fed to the model changes (cached first, fresh after), but in our
     * implementor prompts NO_CACHE is at the end anyway — the real shift is minimal.
     */
    val reorderStrategy: CacheReorderStrategy = GroupByCacheability,
    val growthStrategy: CacheGrowthStrategy = CacheGrowthStrategy.AlwaysGrow,
)

internal fun GeminiCacheConfig.toCacheManagerConfig(): CacheManagerConfig = CacheManagerConfig(
    ttl = ttl,
    refreshDuration = refreshDuration,
    minCacheSizeBytes = minCacheSizeBytes,
    reorderStrategy = reorderStrategy,
    growthStrategy = growthStrategy,
)
