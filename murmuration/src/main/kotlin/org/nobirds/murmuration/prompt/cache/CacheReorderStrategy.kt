package org.nobirds.murmuration.prompt.cache

import org.nobirds.murmuration.internal.logger
import org.nobirds.murmuration.prompt.PromptCacheScope
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptPart

private fun PromptPart.isCacheable(): Boolean = cacheStrategy.scope != PromptCacheScope.NO_CACHE

/**
 * Strategy for choosing which parts go into the cache.
 *
 * Gemini Context Cache (and similar APIs of other providers) mounts cached_content
 * IN FRONT of the provided contents. The cache content need not be a "contiguous prefix
 * of the source prompt" — it is a separate set that is assembled with the fresh tail on the
 * model side. This gives freedom of choice: either a strict prefix (order is preserved), or
 * collect everything cacheable across NO_CACHE "holes" (the order seen by the model changes).
 */
sealed interface CacheReorderStrategy {
    fun collect(content: List<PromptContent>): List<PromptContent>
}

/**
 * Takes a prefix of cacheable parts; stops at the first NO_CACHE part. Everything after
 * it — even if marked PIPELINE — does not get into the cache. Preserves the exact order
 * of the source prompt when presented to the model.
 *
 * If cacheable parts were skipped as a result (there is a NO_CACHE "hole", but PIPELINE/TTL_BASED
 * remained after it) — logs a WARN with the count, so the prompt author can either
 * reorder the parts, or switch to [GroupByCacheability].
 */
data object Strict : CacheReorderStrategy {
    private val logger = logger<Strict>()

    override fun collect(content: List<PromptContent>): List<PromptContent> {
        val head = collectHead(content)
        val taken = head.sumOf { it.parts.count { p -> p.isCacheable() } }
        val total = content.sumOf { it.parts.count { p -> p.isCacheable() } }
        val missed = total - taken
        if (missed > 0) {
            logger.warn(
                "STRICT cache strategy skipped {}/{} cacheable part(s) after a NO_CACHE block. " +
                    "Reorder cacheable parts to come BEFORE NO_CACHE, or switch to GroupByCacheability.",
                missed, total,
            )
        }
        return head
    }

    /** Prefix of cacheable parts up to the first NO_CACHE "hole" (inclusive of [PromptContent]). */
    private fun collectHead(content: List<PromptContent>): List<PromptContent> = buildList {
        for (c in content) {
            val taken = c.parts.takeWhile { it.isCacheable() }
            if (taken.isNotEmpty()) add(PromptContent(c.type, taken))
            if (taken.size < c.parts.size) break
        }
    }
}

/**
 * Collects all cacheable parts across NO_CACHE "holes" (filter, not takeWhile).
 * Cached gets all PIPELINE/TTL_BASED parts in their relative source order,
 * the fresh tail — all NO_CACHE parts. The model sees all cached first, all fresh after.
 * Enable where the author accepts this shift in exchange for maximizing cache hit (typically —
 * implementor prompts, where NO_CACHE is at the end anyway as a "volatile tail").
 */
data object GroupByCacheability : CacheReorderStrategy {
    override fun collect(content: List<PromptContent>): List<PromptContent> = content.mapNotNull { c ->
        val cacheable = c.parts.filter { it.isCacheable() }
        if (cacheable.isEmpty()) null else PromptContent(c.type, cacheable)
    }
}
