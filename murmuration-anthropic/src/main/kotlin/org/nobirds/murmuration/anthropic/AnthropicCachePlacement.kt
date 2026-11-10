package org.nobirds.murmuration.anthropic

import com.anthropic.models.messages.CacheControlEphemeral
import com.anthropic.models.messages.ContentBlockParam
import com.anthropic.models.messages.MessageParam
import com.anthropic.models.messages.TextBlockParam
import com.anthropic.models.messages.Tool
import org.nobirds.murmuration.prompt.PromptCacheScope
import org.nobirds.murmuration.prompt.PromptCacheStrategy
import kotlin.time.Duration
import kotlin.time.Duration.Companion.minutes

/*
 * Materializer of cache points for the Messages API. A prompt part carries a cache *strategy* (a wish: "this part
 * is stable", plus a minimal lifetime); an Anthropic cache *point* means "the whole prefix up to here is stable".
 * The translation is one rule, applied to the request in API order (tools -> system -> messages):
 *
 *  - a point stands only at the end of the CONTINUOUS stable prefix: the first NO_CACHE part ends it, a point is never
 *    put after or inside a hole (that would pay the 1.25x write on every round-trip and never read it back);
 *  - at most [MAX_POINTS] points; parts are never reordered; the top-level automatic caching of the API is not used;
 *  - one TTL class for all points: the longest wish of the covered parts, rounded up to the API class.
 *
 * A prefix below the model's minimum cacheable length is not an error: the API silently skips the point.
 */

/** Hard limit of the API: explicit cache points per request. */
internal const val MAX_POINTS = 4

/** The API searches for an earlier cache entry at most this many blocks before a point. */
internal const val LOOKBACK_BLOCKS = 20

private val SHORT_TTL = 5.minutes

/** Request position the planner sees: one tool, one system block or one message block, in API order. */
internal data class CacheSlot(
    /** Part of the stable prefix: a tool, or a part whose strategy is not NO_CACHE. */
    val stable: Boolean,
    /** The part was explicitly marked cacheable by its author (tools are stable but carry no mark of their own). */
    val marked: Boolean,
    /** Lifetime wished by the part; null — none. */
    val wish: Duration?,
    /** Whether the block type can carry `cache_control` (thinking and redacted thinking blocks cannot). */
    val markable: Boolean,
)

internal enum class CacheTtl { FIVE_MINUTES, ONE_HOUR }

internal data class CachePlan(
    /** Indexes of [CacheSlot]s that receive a point, ascending. */
    val points: List<Int>,
    /** TTL class of every point. */
    val ttl: CacheTtl,
) {
    companion object {
        val NONE = CachePlan(emptyList(), CacheTtl.FIVE_MINUTES)
    }
}

/**
 * Chooses the cache points for [slots].
 *
 * Candidates, by priority: the tail of the stable prefix (the sliding point), the end of
 * the stable system part, the end of the tools; the remaining budget goes to intermediate points that keep every gap within [LOOKBACK_BLOCKS], so the
 * API's lookback from a later point still finds an earlier entry. Nothing is placed unless at least one part of the
 * stable prefix was marked by its author: tools alone are not a reason to pay for a write.
 */
internal fun planCachePoints(slots: List<CacheSlot>, toolCount: Int, systemCount: Int): CachePlan {
    val stableEnd = slots.indexOfFirst { !it.stable }.let { if (it < 0) slots.size else it } // exclusive
    val prefix = slots.subList(0, stableEnd)
    if (prefix.none { it.marked }) return CachePlan.NONE

    fun lastMarkableIn(range: IntRange) = range.lastOrNull { it < stableEnd && slots[it].markable }
    val tail = lastMarkableIn(0 until stableEnd) ?: return CachePlan.NONE
    val chosen = sortedSetOf(tail)
    lastMarkableIn(toolCount until toolCount + systemCount)?.let { if (chosen.size < MAX_POINTS) chosen += it }
    lastMarkableIn(0 until toolCount)?.let { if (chosen.size < MAX_POINTS) chosen += it }

    while (chosen.size < MAX_POINTS) {
        val (prev, next) = chosen.zipWithNext().firstOrNull { (prev, next) -> next - prev > LOOKBACK_BLOCKS } ?: break
        // The latest markable slot that is still a full lookback window before `next`.
        val midpoint = lastMarkableIn((prev + 1)..(next - LOOKBACK_BLOCKS)) ?: break
        if (!chosen.add(midpoint)) break
    }

    val wished = prefix.mapNotNull { it.wish }.maxOrNull()
    val ttl = if (wished != null && wished > SHORT_TTL) CacheTtl.ONE_HOUR else CacheTtl.FIVE_MINUTES
    return CachePlan(chosen.toList(), ttl)
}

private fun CacheTtl.toControl(): CacheControlEphemeral = CacheControlEphemeral.builder()
    .ttl(if (this == CacheTtl.ONE_HOUR) CacheControlEphemeral.Ttl.TTL_1H else CacheControlEphemeral.Ttl.TTL_5M)
    .build()

private fun PromptCacheStrategy.isStable() = scope != PromptCacheScope.NO_CACHE

private fun PromptCacheStrategy.toSlot(markable: Boolean) =
    CacheSlot(stable = isStable(), marked = isStable(), wish = ttl, markable = markable)

private fun ContentBlockParam.canCarryCacheControl() = isText() || isImage() || isToolUse() || isToolResult()

/** The tools, system and messages of one request after cache points were placed. */
internal data class CachedRequest(
    val tools: List<Tool>,
    val system: List<TextBlockParam>,
    val messages: List<MessageParam>,
)

/**
 * Places the cache points on [tools] and this history and returns the request parts with `cache_control` set.
 * Strategies come from the prompt parts ([AnthropicMessages.systemCache], [AnthropicMessages.messageCache]);
 * tools are stable by nature (a tool set does not change inside a request).
 */
internal fun AnthropicMessages.withCachePoints(tools: List<Tool>): CachedRequest {
    val slots = buildList {
        tools.forEach { add(CacheSlot(stable = true, marked = false, wish = null, markable = true)) }
        system.indices.forEach { add(systemCache[it].toSlot(markable = true)) }
        messages.forEachIndexed { m, message ->
            message.content().asBlockParams().forEachIndexed { b, block ->
                add(messageCache[m][b].toSlot(markable = block.canCarryCacheControl()))
            }
        }
    }
    val plan = planCachePoints(slots, toolCount = tools.size, systemCount = system.size)
    if (plan.points.isEmpty()) return CachedRequest(tools, system, messages)

    val control = plan.ttl.toControl()
    val controls = plan.points.associateWith { control }
    var index = 0
    val newTools = tools.map { tool -> tool.toBuilder().apply { controls[index++]?.let { cacheControl(it) } }.build() }
    val newSystem = system.map { block -> block.toBuilder().apply { controls[index++]?.let { cacheControl(it) } }.build() }
    val newMessages = messages.map { message ->
        val blocks = message.content().asBlockParams().map { block ->
            controls[index++]?.let { block.withCacheControl(it) } ?: block
        }
        MessageParam.builder().role(message.role()).contentOfBlockParams(blocks).build()
    }
    return CachedRequest(newTools, newSystem, newMessages)
}

private fun ContentBlockParam.withCacheControl(control: CacheControlEphemeral): ContentBlockParam = when {
    isText() -> ContentBlockParam.ofText(asText().toBuilder().cacheControl(control).build())
    isImage() -> ContentBlockParam.ofImage(asImage().toBuilder().cacheControl(control).build())
    isToolUse() -> ContentBlockParam.ofToolUse(asToolUse().toBuilder().cacheControl(control).build())
    isToolResult() -> ContentBlockParam.ofToolResult(asToolResult().toBuilder().cacheControl(control).build())
    else -> error("A block that cannot carry cache_control was chosen as a cache point")
}
