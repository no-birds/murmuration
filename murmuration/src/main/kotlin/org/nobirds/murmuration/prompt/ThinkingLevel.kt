package org.nobirds.murmuration.prompt

/**
 * Provider-neutral hint for how much "reasoning" the model should generate before answering.
 * Models that don't support a discrete level (only a token budget, or no thinking at all)
 * either ignore the value or use a provider-specific mapping.
 *
 * For Gemini 3 Pro this maps directly to `ThinkingConfig.thinkingLevel`. For Gemini 2.5
 * Flash, where the API expects a numeric `thinkingBudget`, the level is ignored — use
 * [PromptSettings.thinkingBudget] there instead.
 *
 * Pricing-wise the level controls the *count* of reasoning tokens (which are billed as
 * regular output), not a per-token rate.
 */
enum class ThinkingLevel {
    MINIMAL,
    LOW,
    MEDIUM,
    HIGH
}
