package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.prompt.CacheKind
import org.nobirds.murmuration.prompt.PromptCacheStrategy
import kotlin.time.Duration.Companion.minutes

/**
 * Desired lifetime of the agent's marks: a turn runs continuously between round-trips and every read
 * extends the entry, so the short class is enough.
 */
val AGENT_MARK_TTL = 5.minutes

/**
 * Where an agent places cache marks on the history it accumulates, derived solely from the model's
 * [CacheKind]; there is no separate switch, so a tail mark on a pruned history is unrepresentable.
 *
 * | kind     | agent marks on its history |
 * |----------|----------------------------|
 * | MARKS    | sliding tail mark          |
 * | EXECUTOR | none (`NO_CACHE`)          |
 */
enum class CachePlacement(
    /** Cache strategy the agent gives to every part of history it creates itself. */
    val historyCacheStrategy: PromptCacheStrategy,
) {
    /** The agent marks nothing; the executor decides what is cached. */
    NONE(PromptCacheStrategy.NO_CACHE),

    /** All agent-created history is marked stable, so the cacheable prefix reaches the tail. */
    SLIDING_TAIL(PromptCacheStrategy.ttl(AGENT_MARK_TTL)),
    ;

    companion object {
        fun of(kind: CacheKind): CachePlacement = when (kind) {
            CacheKind.MARKS -> SLIDING_TAIL
            CacheKind.EXECUTOR -> NONE
        }
    }
}
