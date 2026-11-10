package org.nobirds.murmuration.llm

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.prompt.*

import kotlinx.serialization.Serializable

/**
 * Token rates of one price level, USD per 1M tokens: the rates of a [PriceTier]. A tier is chosen
 * by the full input of one round-trip; used only by profiles that have tiers (Claude).
 */
@Serializable
data class PriceRates(
    val inputPricePerMillion: Double,
    val cachedInputPricePerMillion: Double,
    val outputPricePerMillion: Double,
    /** Writing into the cache with the short (5 minute) TTL. */
    val cacheWriteShortPricePerMillion: Double,
    /** Writing into the cache with the long (1 hour) TTL. */
    val cacheWriteLongPricePerMillion: Double,
)

/**
 * A price level by prompt length. It applies to a round-trip whose FULL input
 * ([org.nobirds.murmuration.prompt.InputTokenStatistics.total]) is strictly greater than
 * [aboveInputTokens]; exactly [aboveInputTokens] is still the lower level ("up to 100K" is
 * inclusive). The tier with the highest exceeded threshold wins.
 */
@Serializable
data class PriceTier(val aboveInputTokens: Long, val rates: PriceRates) {
    init {
        require(aboveInputTokens >= 0) { "aboveInputTokens must be >= 0, got $aboveInputTokens" }
    }
}

/**
 * Per-model pricing in USD per 1M tokens, attached directly to an [Llm] via
 * [Llm.pricingProfile]. [LlmCostCalculator] looks the [Llm] up by the actual model name
 * through [LlmRegistry] and reads the profile from the matched constant.
 *
 * Cost formula (see [LlmCostCalculator]):
 *   regular_input    = (input_total - input_cached) × inputPricePerMillion / 1M
 *   cached_input     = input_cached × cachedInputPricePerMillion / 1M
 *   cache_creation   = created_cache_tokens × inputPricePerMillion / 1M
 *   cache_storage    = (created_cache_tokens × ttl_hours) × cacheStoragePricePerMillionPerHour / 1M
 *   output           = completion × outputPricePerMillion / 1M
 *   thinking         = thinking × (thinkingPricePerMillion ?: outputPricePerMillion) / 1M
 *   per_request      = request_count × perRequestUsd                (flat-billed models, e.g. Lyria)
 */
@Serializable
data class LlmPricingProfile(
    val inputPricePerMillion: Double,
    val cachedInputPricePerMillion: Double,
    val outputPricePerMillion: Double,
    val cacheStoragePricePerMillionPerHour: Double = 0.0,
    /**
     * Price of thinking tokens, $/1M. `null` → thinking is billed as regular output
     * ([outputPricePerMillion]) — correct for text models, where Gemini bills it that way.
     * For image models output (the picture) is expensive ($60/1M for flash-image), while thinking goes
     * at a separate cheap rate ($3/1M) — the field is set explicitly there, otherwise thinking
     * would be counted at the image price and overstate the cost ~20x. So disabling thinking for
     * image models makes no sense — it costs pennies.
     */
    val thinkingPricePerMillion: Double? = null,
    /**
     * Flat price in USD per generation (one round-trip), for models billed per request rather than
     * per token — e.g. Lyria music generation. `null` for token-billed models. Added on top of any
     * token costs (typically zero for such models).
     */
    val perRequestUsd: Double? = null,
    /**
     * Cache-write prices, $/1M, by TTL class: `null` for flat profiles (Gemini, OpenAI/Azure), which
     * bill no separate cache write. Both are set together, only through [withCacheWrite].
     */
    val cacheWriteShortPricePerMillion: Double? = null,
    val cacheWriteLongPricePerMillion: Double? = null,
    /** Levels above the base rates, by full round-trip input; see [PriceTier]. Empty — flat pricing. */
    val tiers: List<PriceTier> = emptyList(),
) {
    companion object {
        /**
         * A profile of a provider that bills cache writes (Anthropic): both TTL write prices are
         * required, so "write billed as regular input" cannot be expressed.
         */
        fun withCacheWrite(
            inputPricePerMillion: Double,
            cachedInputPricePerMillion: Double,
            outputPricePerMillion: Double,
            cacheWriteShortPricePerMillion: Double,
            cacheWriteLongPricePerMillion: Double,
            tiers: List<PriceTier> = emptyList(),
            thinkingPricePerMillion: Double? = null,
        ): LlmPricingProfile = LlmPricingProfile(
            inputPricePerMillion = inputPricePerMillion,
            cachedInputPricePerMillion = cachedInputPricePerMillion,
            outputPricePerMillion = outputPricePerMillion,
            thinkingPricePerMillion = thinkingPricePerMillion,
            cacheWriteShortPricePerMillion = cacheWriteShortPricePerMillion,
            cacheWriteLongPricePerMillion = cacheWriteLongPricePerMillion,
            tiers = tiers,
        )
    }

    /** Effective thinking price: explicit [thinkingPricePerMillion], or the output price (text default). */
    val effectiveThinkingPricePerMillion: Double
        get() = thinkingPricePerMillion ?: outputPricePerMillion
}
