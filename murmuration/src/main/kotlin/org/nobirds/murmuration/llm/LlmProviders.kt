package org.nobirds.murmuration.llm

import org.nobirds.murmuration.prompt.CacheKind

/** Provider identifiers the core needs to recognise. */
object LlmProviders {
    const val ANTHROPIC = "anthropic"
}

/** Cache kind of this model: `anthropic` provider ⇒ [CacheKind.MARKS], any other ⇒ [CacheKind.EXECUTOR]. */
val Llm.cacheKind: CacheKind
    get() = if (provider == LlmProviders.ANTHROPIC) CacheKind.MARKS else CacheKind.EXECUTOR
