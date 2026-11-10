package org.nobirds.murmuration.prompt.cache

import kotlinx.coroutines.CoroutineScope
import org.nobirds.murmuration.agent.PipelineEventListener
import org.nobirds.murmuration.internal.logger
import org.nobirds.murmuration.llm.Caching
import org.nobirds.murmuration.llm.capability
import org.nobirds.murmuration.prompt.InputTokenStatistics
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptCacheScope
import org.nobirds.murmuration.prompt.PromptCachingProvider
import org.nobirds.murmuration.prompt.PromptCachingProvider.CacheResult
import org.nobirds.murmuration.prompt.PromptCachingProvider.CacheStatistics
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContextCacheEntry
import org.nobirds.murmuration.prompt.PromptContextCacheStorage
import org.nobirds.murmuration.prompt.PromptExecutionStatistics
import org.nobirds.murmuration.prompt.PromptExecutionTokenStatistics
import org.nobirds.murmuration.prompt.PromptPart
import org.nobirds.murmuration.prompt.toMessage
import kotlin.time.Clock
import kotlin.time.Duration
import kotlin.time.Instant
import kotlin.time.measureTimedValue

/**
 * Backend-agnostic prompt cache manager. Implements `PromptCachingProvider`, delegating
 * provider-specific operations to `CacheBackend`. The logic lives here and only here:
 *  - Which parts go into the cache (via `CacheReorderStrategy`).
 *  - How to look up an existing prefix cache in storage.
 *  - When to extend the TTL.
 *  - When to grow a long cache on top of a found prefix (`CacheGrowthStrategy`).
 *  - How to "cut out" cached parts from the fresh prompt.
 *
 * `CacheBackend` is responsible only for calls to the specific provider API and content hashing.
 */
class CacheManager(
    private val backend: CacheBackend,
    private val storage: PromptContextCacheStorage,
    private val config: CacheManagerConfig = CacheManagerConfig(),
) : PromptCachingProvider {
    private val logger = logger()

    override fun createPipelineListener(
        scope: CoroutineScope,
        emitter: PipelineEventListener,
    ): PipelineEventListener =
        CachePipelineCleaner(scope, storage, backend, emitter)

    override suspend fun processPrompt(prompt: Prompt): CacheResult {
        val cacheableContents = config.reorderStrategy.collect(prompt.content)

        if (cacheableContents.isEmpty()) {
            return CacheResult(prompt = prompt)
        }

        val allCacheableParts = cacheableContents.flatMap { it.parts }
        val fullHash = backend.hashContent(cacheableContents, prompt.settings)
        val model = prompt.llm.model

        val scope = if (allCacheableParts.any { it.cacheStrategy.scope == PromptCacheScope.PIPELINE }) {
            PromptCacheScope.PIPELINE
        } else {
            PromptCacheScope.TTL_BASED
        }

        val strategyTtl = allCacheableParts.mapNotNull { it.cacheStrategy.ttl }.minOrNull()
        val ttl = strategyTtl ?: config.ttl

        val match = findLongestUsableCache(cacheableContents, prompt, fullHash, ttl)
        if (match != null) {
            val (entry, matchedContents) = match
            val now = Clock.System.now()
            val totalSize = allCacheableParts.sumOf { backend.estimateSize(it) }
            val matchedSize = matchedContents.flatMap { it.parts }.sumOf { backend.estimateSize(it) }

            val shouldGrow = matchedContents.size != cacheableContents.size &&
                config.growthStrategy.shouldGrow(matchedSize, totalSize, config.minCacheSizeBytes, entry)

            if (!shouldGrow) {
                logger.debug(
                    "Reusing cache: {} for hash: {} (match size: {}, total: {}, grow: {})",
                    entry.cacheId, entry.contentHash, matchedSize, totalSize, shouldGrow,
                )
                return buildResult(
                    prompt, entry.cacheId, matchedContents, entry.tokenCount,
                    isCreated = false, ttl = entry.expirationTime - now, scope = entry.scope,
                )
            }

            logger.info(
                "Found prefix cache {} (size: {}), but remaining is large ({}). Creating a new longer cache.",
                entry.cacheId, matchedSize, totalSize - matchedSize,
            )
        }

        // Provider rejects cache.create below a per-model TOKEN minimum (Vertex Gemini 3.x: 4096;
        // 2.5: 1024) — declared via the [Caching] capability. We only have a byte estimate here (no
        // tokenizer pre-call), so approximate tokens at a conservative ~4 bytes/token and skip when
        // clearly below, to avoid the wasted ~1.5s round-trip. The residual (estimate too high) is
        // caught below and logged benignly. Models without [Caching] fall back to the flat byte floor.
        val totalCacheableSize = allCacheableParts.sumOf { backend.estimateSize(it) }
        val minBytes = prompt.llm.capability<Caching>()?.let { it.minTokens * 4 } ?: config.minCacheSizeBytes
        if (totalCacheableSize < minBytes) {
            logger.debug("Skipping cache.create — content {} bytes below minimum {} bytes.", totalCacheableSize, minBytes)
            return CacheResult(prompt = prompt)
        }

        return try {
            val (created, createDuration) = measureTimedValue {
                backend.createCache(model, cacheableContents, prompt.settings, fullHash, ttl, scope)
            }
            if (created.cacheId.isEmpty()) return CacheResult(prompt = prompt)
            storage.save(
                PromptContextCacheEntry(
                    cacheId = created.cacheId,
                    contentHash = fullHash,
                    model = model,
                    scope = scope,
                    expirationTime = created.expirationTime,
                    createTime = Clock.System.now(),
                    tokenCount = created.tokenCount,
                )
            )
            buildResult(
                prompt, created.cacheId, cacheableContents, created.tokenCount,
                isCreated = true, ttl = ttl, scope = scope,
                createStatistics = buildCreateStatistics(model, prompt, cacheableContents, created.cacheId, created.tokenCount, ttl, scope),
                createDuration = createDuration,
            )
        } catch (e: Exception) {
            // A sub-minimum cache.create can slip past the byte estimate above — benign (we just skip
            // caching), so log it quietly; other failures stay WARN with the stack.
            if (e.message?.contains("minimum token count", ignoreCase = true) == true) {
                logger.debug("Skipping cache.create — content below provider token minimum: {}", e.message)
            } else {
                logger.warn("Failed to create cache: ${e.message}", e)
            }
            CacheResult(prompt = prompt)
        }
    }

    /**
     * Builds [PromptExecutionStatistics] for a CACHE_CREATE row. `requestMessages` is the full
     * cache content (as sent to `caches.create`); this is "what lies in cache N" for
     * post-analysis of prompts. Cost is not counted as a separate row here — exactly as with a
     * round-trip row with `isCacheCreated == true`, so that the calculator does not double-count
     * creation cost.
     */
    private fun buildCreateStatistics(
        model: String,
        prompt: Prompt,
        cacheableContents: List<PromptContent>,
        cacheId: String,
        tokenCount: Long,
        ttl: Duration,
        scope: PromptCacheScope,
    ): PromptExecutionStatistics = PromptExecutionStatistics(
        modelName = model,
        requestedModelName = prompt.llm.model,
        attempts = 1,
        success = true,
        tokens = PromptExecutionTokenStatistics(
            totalTokens = tokenCount,
            input = InputTokenStatistics(total = tokenCount, cached = 0, cacheWriteShort = 0, cacheWriteLong = 0),
        ),
        requestMessages = cacheableContents.map { it.toMessage() },
        responseMessage = null,
        cacheId = cacheId,
        cachedTokenCount = tokenCount,
        isCacheCreated = true,
        cacheTtlMillis = ttl.inWholeMilliseconds,
        cacheScope = scope,
    )

    override suspend fun keepAlive(cacheId: String): Instant? {
        val entry = storage.findById(cacheId) ?: run {
            logger.warn("Keep-alive: cache $cacheId not found in storage — stopping underwriter")
            return null
        }
        val now = Clock.System.now()
        val remaining = entry.expirationTime - now
        return when {
            remaining <= Duration.ZERO -> {
                logger.warn("Keep-alive: cache $cacheId already expired (remaining=$remaining) — stopping underwriter")
                null
            }
            remaining > config.refreshDuration -> {
                // TTL is still healthy — refreshing is premature, don't even hit the API.
                entry.expirationTime
            }
            else -> {
                try {
                    val newExpiration = backend.refreshCache(cacheId, config.ttl)
                    storage.updateExpiration(cacheId, newExpiration)
                    logger.debug("Keep-alive: extended cache {} (was remaining={}), new expiration: {}", cacheId, remaining, newExpiration)
                    newExpiration
                } catch (e: Exception) {
                    logger.warn("Keep-alive: failed to refresh cache $cacheId: ${e.message}")
                    // Return the current expiration — the next iteration may cope.
                    entry.expirationTime
                }
            }
        }
    }

    /**
     * Looks in storage for the longest prefix of the current set of cacheable contents with a live TTL.
     * If the found one's TTL is about to end — tries to extend it via the backend; if that fails —
     * we treat it as not found. Returns `null` if no prefix fits.
     *
     * For `TTL_BASED` caches the TTL is **sliding**: EACH use extends the cache by the
     * full [ttl] (the strategy TTL of the current prompt). Without this the cache died exactly TTL after
     * CREATION even when actively hit: the narrow `refreshDuration` window almost never
     * matched the turn cadence (live example: a turn reused a cache with 8 min left — without
     * extension, and the next turn was 75 seconds late for it and paid for a full re-creation).
     */
    private suspend fun findLongestUsableCache(
        cacheableContents: List<PromptContent>,
        prompt: Prompt,
        fullHash: String,
        ttl: Duration,
    ): Pair<PromptContextCacheEntry, List<PromptContent>>? {
        for (i in cacheableContents.size downTo 1) {
            val subContents = cacheableContents.take(i)
            val subHash = if (i == cacheableContents.size) fullHash else backend.hashContent(subContents, prompt.settings)
            val existing = storage.findByHash(subHash, prompt.llm.model) ?: continue

            val now = Clock.System.now()
            val remaining = existing.expirationTime - now
            when {
                remaining <= Duration.ZERO -> continue // expired — skip
                existing.scope == PromptCacheScope.TTL_BASED && remaining < ttl -> {
                    val refreshed = try {
                        val newExpiration = backend.refreshCache(existing.cacheId, ttl)
                        storage.updateExpiration(existing.cacheId, newExpiration)
                        logger.debug(
                            "Slid TTL_BASED cache {} to +{} (was remaining {})",
                            existing.cacheId, ttl, remaining,
                        )
                        existing.copy(expirationTime = newExpiration)
                    } catch (e: Exception) {
                        logger.warn("Failed to slide TTL for cache ${existing.cacheId}: ${e.message}")
                        // The cache is still alive — usable without extension only if not on the verge of death.
                        if (remaining >= config.refreshDuration) existing else null
                    }
                    if (refreshed != null) return refreshed to subContents
                }
                remaining < config.refreshDuration -> {
                    val refreshed = try {
                        val newExpiration = backend.refreshCache(existing.cacheId, config.refreshDuration)
                        storage.updateExpiration(existing.cacheId, newExpiration)
                        logger.info(
                            "Refreshed cache {} (remaining: {} < {}), new expiration: {}",
                            existing.cacheId, remaining, config.refreshDuration, newExpiration,
                        )
                        existing.copy(expirationTime = newExpiration)
                    } catch (e: Exception) {
                        logger.warn("Failed to refresh cache ${existing.cacheId}: ${e.message}")
                        null
                    }
                    if (refreshed != null) return refreshed to subContents
                }
                else -> return existing to subContents
            }
        }
        return null
    }

    /**
     * Identity-based "cutting out" of cached parts from the source prompt. With `GROUP_BY_CACHEABILITY`
     * the cached parts are NOT a contiguous prefix, so dropping by index does not work.
     * `IdentityHashMap` is critical — two data-equal parts (same text and cacheStrategy)
     * must be distinguishable if only one of them ended up in the cache.
     */
    private fun buildResult(
        prompt: Prompt,
        cacheId: String,
        cachedContents: List<PromptContent>,
        tokenCount: Long,
        isCreated: Boolean,
        ttl: Duration? = null,
        scope: PromptCacheScope? = null,
        createStatistics: PromptExecutionStatistics? = null,
        createDuration: Duration? = null,
    ): CacheResult {
        val cachedParts: MutableSet<PromptPart> =
            java.util.Collections.newSetFromMap(java.util.IdentityHashMap())
        cachedParts.addAll(cachedContents.flatMap { it.parts })

        val newContent = prompt.content.mapNotNull { content ->
            val remaining = content.parts.filterNot { it in cachedParts }
            if (remaining.isEmpty()) null else PromptContent(content.type, remaining)
        }

        return CacheResult(
            id = cacheId,
            prompt = Prompt(prompt.llm, prompt.settings, newContent),
            statistics = CacheStatistics(
                tokenCount = tokenCount,
                isCreated = isCreated,
                ttl = ttl,
                scope = scope,
            ),
            createStatistics = createStatistics,
            createDuration = createDuration,
        )
    }
}
