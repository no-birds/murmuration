package org.nobirds.murmuration.prompt.cache

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import org.nobirds.murmuration.agent.AgentExecutionEntry
import org.nobirds.murmuration.agent.ExecutionNature
import org.nobirds.murmuration.agent.PipelineEventListener
import org.nobirds.murmuration.agent.PromptExecutionEntry
import org.nobirds.murmuration.internal.logger
import org.nobirds.murmuration.prompt.PromptCacheScope
import org.nobirds.murmuration.prompt.PromptContextCacheStorage
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import kotlin.time.Clock
import kotlin.time.Instant

/**
 * Listener that deletes all `PromptCacheScope.PIPELINE` caches after the corresponding
 * pipeline finishes. Backend-agnostic: the actual deletion is delegated to `CacheBackend.deleteCache`.
 *
 * Listens to CACHE_CREATE rows (`PromptExecutionEntry` with `nature = CACHE_CREATE`) — each
 * created pipeline cache has exactly one such row, from which we take `cacheId` and `cacheScope`.
 * After a successful deletion it reports [PipelineEventListener.onCacheDeleted] to [emitter] —
 * a composite of other listeners, including the application-side writer, which records the actual
 * cache lifetime in `llm_statistics.cache_lifetime_millis` for exact storage billing.
 *
 * Deletion is tied to the RECORD, not to the step that created the cache: a run composed of several
 * steps keeps every PIPELINE-scoped cache alive until the whole run finishes, so its storage is
 * billed for that longer span.
 */
internal class CachePipelineCleaner(
    private val scope: CoroutineScope,
    private val storage: PromptContextCacheStorage,
    private val backend: CacheBackend,
    private val emitter: PipelineEventListener,
) : PipelineEventListener {
    private val logger = logger()
    private val cachesToDelete = ConcurrentHashMap<UUID, MutableMap<String, Instant>>()

    override suspend fun onEntryReported(pipelineId: UUID, entry: AgentExecutionEntry) {
        if (entry !is PromptExecutionEntry) return
        if (entry.nature != ExecutionNature.CACHE_CREATE) return
        val stats = entry.statistics
        if (stats.cacheScope != PromptCacheScope.PIPELINE) return
        val cacheId = stats.cacheId ?: return
        cachesToDelete.getOrPut(pipelineId) { ConcurrentHashMap() }
            .putIfAbsent(cacheId, Clock.System.now())
    }

    override suspend fun onFinished(pipelineId: UUID) {
        cleanupCaches(pipelineId)
    }

    /**
     * Pipeline failures go through `onError` (see the `executePipelineWithBudget` finally block:
     * on success → onFinished, on failure → onError, **not both at once**). Without handling the failure path
     * pipeline caches stay alive in Vertex until their TTL — the next run may pick up
     * a foreign cache of a previous failed run. Cleanup is idempotent (deleteCache is wrapped inside
     * try/catch for cache-expired) — safe to call on both paths.
     */
    override suspend fun onError(pipelineId: UUID, throwable: Throwable) {
        cleanupCaches(pipelineId)
    }

    private fun cleanupCaches(pipelineId: UUID) {
        val caches = cachesToDelete.remove(pipelineId) ?: return
        scope.launch {
            caches.forEach { (cacheId, createdAt) ->
                try {
                    logger.info("Deleting pipeline cache: $cacheId for pipeline: $pipelineId")
                    backend.deleteCache(cacheId)
                    storage.delete(cacheId)
                    // coerce to 0: if the wall clock jumps back (now < createdAt) the difference would go
                    // negative and leak into totalUsd as a negative storage cost. We clamp at the source.
                    val lifetimeMillis = (Clock.System.now() - createdAt).inWholeMilliseconds.coerceAtLeast(0)
                    logger.debug("Successfully deleted cache: $cacheId (lived ${lifetimeMillis}ms)")
                    emitter.onCacheDeleted(pipelineId, cacheId, lifetimeMillis)
                } catch (e: Exception) {
                    logger.warn("Failed to delete pipeline cache: $cacheId. It might have been already deleted or expired. Error: ${e.message}")
                }
            }
        }
    }
}
