package org.nobirds.murmuration.prompt.cache

import org.nobirds.murmuration.prompt.PromptCacheScope
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptPart
import org.nobirds.murmuration.prompt.PromptSettings
import kotlin.time.Duration
import kotlin.time.Instant

/**
 * "Provider specifics" contract for `CacheManager`. Implementations call the API
 * of a particular LLM provider (Gemini Context Cache, Anthropic Prompt Caching, etc.)
 * and convert our `PromptPart`s into the provider's native format.
 *
 * The manager talks to the backend ONLY through this interface — no Gemini/Anthropic
 * detail may leak into `CacheManager`.
 */
interface CacheBackend {
    /**
     * Creates a new cache on the provider side and returns its parameters. The manager itself
     * decides when to call it (after the storage lookup), the backend just executes.
     */
    suspend fun createCache(
        model: String,
        contents: List<PromptContent>,
        settings: PromptSettings?,
        hash: String,
        ttl: Duration,
        scope: PromptCacheScope,
    ): CreateCacheResult

    /**
     * Extends the TTL of an existing cache. Returns the new expiration time.
     */
    suspend fun refreshCache(cacheId: String, ttl: Duration): Instant

    /**
     * Deletes the cache on the provider side. The manager cleans storage separately.
     */
    suspend fun deleteCache(cacheId: String)

    /**
     * Stable content hash for storage lookup. Depends on the backend because the
     * semantics of "these parts are equivalent for model X" may differ between providers
     * (for example, tools are serialized differently in Gemini than in Anthropic).
     */
    fun hashContent(contents: List<PromptContent>, settings: PromptSettings?): String

    /**
     * Rough estimate of a part's "weight" in bytes — for comparison with `minCacheSizeBytes` and deciding
     * whether "it is worth creating a new long cache". An implementation may count text characters,
     * a nominal file size, etc.
     */
    fun estimateSize(part: PromptPart): Int
}

data class CreateCacheResult(
    val cacheId: String,
    val tokenCount: Long,
    val expirationTime: Instant,
)
