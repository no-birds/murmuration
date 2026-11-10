package org.nobirds.murmuration.anthropic

import com.anthropic.models.messages.OutputConfig
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.llm.LlmPricingProfile
import org.nobirds.murmuration.llm.LlmProviders
import org.nobirds.murmuration.llm.PriceRates
import org.nobirds.murmuration.llm.PriceTier
import org.nobirds.murmuration.llm.llm
import org.nobirds.murmuration.prompt.ThinkingLevel

/**
 * An Anthropic model: the [Llm] constant (registered in `LlmRegistry` with its pricing) plus
 * the provider-side constants the core abstraction has no place for.
 *
 * [minimalThinking] is the least thinking this model can be asked for (see [AnthropicThinking]);
 * only the provider registry knows it, the core and application code never do.
 */
data class AnthropicModelSpec(
    val llm: Llm,
    val contextTokens: Int,
    val maxOutputTokens: Int,
    val minimalThinking: AnthropicThinking,
)

/**
 * Model catalogue of the Anthropic provider (Claude 5.5 family), mirror of `OpenAiLlms` /
 * `GeminiLlms`: [ALL] is registered in `LlmRegistry` at application startup, [findModel] resolves a
 * model by provider and actual model name. Prices: USD per 1M tokens, checked 2026-10-25. Only Haiku has a price tier (100K of the full
 * input of one round-trip); Sonnet and Opus are flat over the whole 1M window. Thinking levels: the explicit set
 * [THINKING_LEVELS] on every model (all four core levels are expressible: MINIMAL is the model's minimal form, see [AnthropicThinking]).
 */
object AnthropicLlms {

    const val PROVIDER = LlmProviders.ANTHROPIC
    private const val CONTEXT_TOKENS = 1_000_000
    private const val MAX_OUTPUT_TOKENS = 128_000

    /** Core thinking levels a Claude 5.5 model accepts: MINIMAL → the model's minimal form, LOW/MEDIUM/HIGH → adaptive + effort. */
    private val THINKING_LEVELS = arrayOf(ThinkingLevel.MINIMAL, ThinkingLevel.LOW, ThinkingLevel.MEDIUM, ThinkingLevel.HIGH)

    private val OPUS_5_5_PRICING = LlmPricingProfile.withCacheWrite(
        inputPricePerMillion = 4.0,
        cachedInputPricePerMillion = 0.20,
        outputPricePerMillion = 20.0,
        cacheWriteShortPricePerMillion = 5.0,
        cacheWriteLongPricePerMillion = 8.0,
    )

    // Sonnet read is 0.10, not 0.20 (the claude-api skill is wrong here).
    private val SONNET_5_5_PRICING = LlmPricingProfile.withCacheWrite(
        inputPricePerMillion = 2.0,
        cachedInputPricePerMillion = 0.10,
        outputPricePerMillion = 10.0,
        cacheWriteShortPricePerMillion = 2.50,
        cacheWriteLongPricePerMillion = 4.0,
    )

    // Haiku is tiered by the FULL input of a round-trip: up to 100K inclusive — base, above — upper tier.
    private val HAIKU_5_5_PRICING = LlmPricingProfile.withCacheWrite(
        inputPricePerMillion = 0.10,
        cachedInputPricePerMillion = 0.01,
        outputPricePerMillion = 0.50,
        cacheWriteShortPricePerMillion = 0.125,
        cacheWriteLongPricePerMillion = 0.20,
        tiers = listOf(
            PriceTier(
                aboveInputTokens = 100_000,
                rates = PriceRates(
                    inputPricePerMillion = 0.50,
                    cachedInputPricePerMillion = 0.05,
                    outputPricePerMillion = 2.50,
                    cacheWriteShortPricePerMillion = 0.625,
                    cacheWriteLongPricePerMillion = 1.00,
                ),
            ),
        ),
    )

    private fun claude(
        displayName: String,
        model: String,
        pricingProfile: LlmPricingProfile,
        minimalThinking: AnthropicThinking,
    ): AnthropicModelSpec = AnthropicModelSpec(
        llm = llm(PROVIDER, model) {
            family = "claude"
            name = displayName
            version = "5.5"
            stable = true
            pricing = pricingProfile
            input { text(); image() }
            output { text(); thoughts() }
            caching(minTokens = 512)
            thinking { level(*THINKING_LEVELS) }
            toolCalling()
        },
        contextTokens = CONTEXT_TOKENS,
        maxOutputTokens = MAX_OUTPUT_TOKENS,
        minimalThinking = minimalThinking,
    )

    val CLAUDE_OPUS_5_5 =
        claude("Claude Opus 5.5", "claude-opus-5-5", OPUS_5_5_PRICING, AnthropicThinking.BetweenTools)

    val CLAUDE_SONNET_5_5 =
        claude("Claude Sonnet 5.5", "claude-sonnet-5-5", SONNET_5_5_PRICING, AnthropicThinking.BetweenTools)

    // Haiku: `disabled` is legal (200) but "thinking off" does not exist in the product model, and
    // `between_tools` was never probed on Haiku. The probed legal minimum is `adaptive` + LOW.
    val CLAUDE_HAIKU_5_5 = claude(
        "Claude Haiku 5.5", "claude-haiku-5-5", HAIKU_5_5_PRICING, AnthropicThinking.Adaptive(OutputConfig.Effort.LOW),
    )

    /** All Anthropic model specs. `by lazy` for the same init-order reason as `GeminiLlms.ALL`. */
    val SPECS: List<AnthropicModelSpec> by lazy { listOf(CLAUDE_OPUS_5_5, CLAUDE_SONNET_5_5, CLAUDE_HAIKU_5_5) }

    /** All Anthropic [Llm] constants, for `LlmRegistry.register`. */
    val ALL: List<Llm> by lazy { SPECS.map { it.llm } }

    /** Resolves a spec by provider and actual model name (longest-[Llm.model] prefix match, as the neighbours). */
    fun findSpec(provider: String, model: String): AnthropicModelSpec =
        SPECS.filter { it.llm.provider == provider && it.llm.matches(model) }
            .maxByOrNull { it.llm.model.length }
            ?: throw IllegalArgumentException("LLM model $model not found")

    /** Resolves an [Llm] constant by provider and actual model name. */
    fun findModel(provider: String, model: String): Llm = findSpec(provider, model).llm
}
