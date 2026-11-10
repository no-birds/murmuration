package org.nobirds.murmuration.prompt

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.llm.*

import kotlin.time.Instant

/**
 * Framework-side DTO for a context-cache entry. Carries no infrastructural details (no
 * surrogate `id` — the natural key is `cacheId`). Each [PromptContextCacheStorage]
 * implementation decides how to persist this physically.
 */
data class PromptContextCacheEntry(
    val cacheId: String,
    val contentHash: String,
    val model: String,
    val scope: PromptCacheScope,
    val expirationTime: Instant,
    val createTime: Instant,
    val tokenCount: Long
)

/**
 * SPI for a context-cache storage. Used by [PromptCachingProvider] implementations that
 * need to persist factual cache state across processes (for example,
 * `GeminiPromptCachingProvider`).
 *
 * Application code is expected to supply an implementation backed by its own database
 * (Exposed, JPA, etc.); the framework itself makes no assumptions about the backing store.
 */
interface PromptContextCacheStorage {
    suspend fun findByHash(hash: String, model: String): PromptContextCacheEntry?
    suspend fun findById(cacheId: String): PromptContextCacheEntry?
    suspend fun save(entry: PromptContextCacheEntry)
    suspend fun updateExpiration(cacheId: String, expirationTime: Instant)
    suspend fun delete(cacheId: String)
}
