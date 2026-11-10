package org.nobirds.murmuration.openai

import com.openai.models.responses.ResponseUsage
import org.nobirds.murmuration.prompt.InputTokenStatistics
import org.nobirds.murmuration.prompt.OutputTokenStatistics
import org.nobirds.murmuration.prompt.PromptExecutionEvent.UsageDelta

/**
 * Maps Azure OpenAI Responses [ResponseUsage] into a provider-neutral [UsageDelta].
 *
 * Input: `inputTokens` is the full input (already includes the cached portion);
 * `inputTokensDetails.cachedTokens` is the cached subset.
 *
 * Output — **note the difference from Gemini**: OpenAI's `outputTokens` **includes** reasoning
 * tokens (Gemini reports `candidatesTokenCount` separately from `thoughtsTokenCount`).
 * [org.nobirds.murmuration.llm.LlmCostCalculator] bills `output.total` and `output.thinking` as
 * two separate summed lines, so to avoid double-counting reasoning we split them here:
 * `output.total = outputTokens − reasoningTokens`, `output.thinking = reasoningTokens`.
 */
internal fun ResponseUsage.toUsageDelta(): UsageDelta {
    val cached = inputTokensDetails().cachedTokens()
    val reasoning = outputTokensDetails().reasoningTokens()
    val input = InputTokenStatistics(total = inputTokens(), cached = cached, cacheWriteShort = 0, cacheWriteLong = 0)
    // outputTokens already includes reasoning — subtract it out of `total` so the cost
    // calculator's separate `output.total` + `output.thinking` sum doesn't count it twice.
    val output = OutputTokenStatistics(total = outputTokens() - reasoning, thinking = reasoning)
    return UsageDelta(input = input, output = output)
}

