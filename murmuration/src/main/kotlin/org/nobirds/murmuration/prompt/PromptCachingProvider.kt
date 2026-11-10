package org.nobirds.murmuration.prompt

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.llm.*

import kotlinx.coroutines.CoroutineScope
import kotlin.time.Duration
import kotlin.time.Instant

/**
 * Interface for providing cached content for prompts.
 */
interface PromptCachingProvider {
    /**
     * Creates a pipeline event listener that can track and manage caches for a specific pipeline.
     * [emitter] is the composite of all pipeline listeners — the cache subsystem uses it to fan
     * out `onCacheDeleted` events after cleaning up pipeline-scoped caches, so application-side writers
     * can update `llm_statistics` with the actual cache lifetime.
     */
    fun createPipelineListener(
        scope: CoroutineScope,
        emitter: PipelineEventListener,
    ): PipelineEventListener? = null

    data class CacheStatistics(
        val tokenCount: Long? = null,
        val isCreated: Boolean = false,
        val ttl: Duration? = null,
        val isExpired: Boolean = false,
        val expiredAt: Instant? = null,
        val scope: PromptCacheScope? = null
    )

    data class CacheResult(
        val id: String? = null,
        val prompt: Prompt,
        val statistics: CacheStatistics? = null,
        /**
         * Ready statistics for a CACHE_CREATE row in `llm_statistics`. Filled only
         * when this call performed an actual `caches.create` (not a reuse). `requestMessages`
         * = full cache content; tokens go to input.total. Passed up through the executor
         * as [PromptExecutionEvent.CacheCreated]; the agent reports it as a separate row.
         */
        val createStatistics: PromptExecutionStatistics? = null,
        /**
         * Duration of the `caches.create` call (cache creation time, excluding the main
         * round-trip). Goes to [PromptExecutionEvent.CacheCreated.duration].
         */
        val createDuration: Duration? = null,
    )

    /**
     * Process the prompt to identify cacheable parts, possibly creating or reusing a Gemini Context Cache.
     * Returns the name of the cache to use (if any) and the modified prompt with cached parts removed.
     */
    suspend fun processPrompt(prompt: Prompt): CacheResult

    /**
     * Hook for a background "underwriter": inspect the cache state and extend its TTL when
     * it is about to expire. The implementation decides for itself whether the extension
     * API actually needs to be called (based on the remaining TTL).
     *
     * Returns the current `expirationTime` of the cache (after the possible extension), or
     * `null` when the cache has already expired / been deleted / cannot be found. On
     * `null`, the underwriter stops further attempts.
     */
    suspend fun keepAlive(cacheId: String): Instant? = null
}

/**
 * A no-op implementation of [PromptCachingProvider] that does nothing and always returns the original prompt.
 */
object NoOpPromptCachingProvider : PromptCachingProvider {
    override suspend fun processPrompt(prompt: Prompt): PromptCachingProvider.CacheResult {
        return PromptCachingProvider.CacheResult(prompt = prompt)
    }
}
