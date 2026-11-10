package org.nobirds.murmuration.openai

import com.openai.models.responses.ResponseUsage
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals

/**
 * Unit tests for [ResponseUsage.toUsageDelta] — the Azure OpenAI Responses usage ->
 * provider-neutral [org.nobirds.murmuration.prompt.PromptExecutionEvent.UsageDelta] mapper. The
 * key behaviour under test is the reasoning-token split: OpenAI's `outputTokens` INCLUDES
 * reasoning tokens (unlike Gemini, which reports them separately), so the mapper must subtract
 * them out of `output.total` to avoid [org.nobirds.murmuration.llm.LlmCostCalculator]
 * double-billing.
 */
class OpenAiUsageMappingTest {

    private fun usage(
        inputTokens: Long,
        outputTokens: Long,
        cachedTokens: Long = 0,
        reasoningTokens: Long = 0,
    ): ResponseUsage = ResponseUsage.builder()
        .inputTokens(inputTokens)
        .outputTokens(outputTokens)
        .totalTokens(inputTokens + outputTokens)
        .inputTokensDetails(ResponseUsage.InputTokensDetails.builder().cachedTokens(cachedTokens).cacheWriteTokens(0).build())
        .outputTokensDetails(ResponseUsage.OutputTokensDetails.builder().reasoningTokens(reasoningTokens).build())
        .build()

    @Test
    fun `input tokens carry the cached subset separately from the total`() {
        val delta = usage(inputTokens = 120, outputTokens = 0, cachedTokens = 50).toUsageDelta()
        assertEquals(120L, delta.input.total)
        assertEquals(50L, delta.input.cached)
    }

    @Test
    fun `reasoning tokens are split out of the output total to avoid double counting`() {
        val delta = usage(inputTokens = 10, outputTokens = 100, reasoningTokens = 30).toUsageDelta()
        assertEquals(70L, delta.output.total)
        assertEquals(30L, delta.output.thinking)
    }

    @Test
    fun `zero details default to zero`() {
        val delta = usage(inputTokens = 10, outputTokens = 5).toUsageDelta()
        assertEquals(10L, delta.input.total)
        assertEquals(0L, delta.input.cached)
        assertEquals(5L, delta.output.total)
        assertEquals(0L, delta.output.thinking)
    }
}
