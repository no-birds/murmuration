package org.nobirds.murmuration.anthropic

import org.nobirds.murmuration.llm.CostCategory
import com.anthropic.models.messages.OutputConfig
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.LlmCostCalculator
import org.nobirds.murmuration.llm.LlmProviders
import org.nobirds.murmuration.llm.Thinking
import org.nobirds.murmuration.llm.ThinkingMode
import org.nobirds.murmuration.llm.capability
import org.nobirds.murmuration.llm.clampThinkingLevel
import org.nobirds.murmuration.llm.RoundTripUsage
import org.nobirds.murmuration.prompt.InputTokenStatistics
import org.nobirds.murmuration.prompt.ThinkingLevel
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

class AnthropicLlmsTest {

    private fun cost(model: String, total: Long, cached: Long = 0, write5m: Long = 0, write1h: Long = 0, output: Long = 0) =
        LlmCostCalculator.roundTrip(
            RoundTripUsage(InputTokenStatistics(total, cached, write5m, write1h), output = output),
            AnthropicLlms.findModel("anthropic", model).pricingProfile,
        )

    @Test
    fun `opus prices regular, output, writes and read`() {
        val c = cost("claude-opus-5-5", total = 4_000_000, cached = 1_000_000, write5m = 1_000_000, write1h = 1_000_000, output = 1_000_000)
        assertEquals(1.0 * 4.0, c[CostCategory.REGULAR_INPUT], 1e-9)
        assertEquals(0.20, c[CostCategory.CACHED_INPUT], 1e-9)
        assertEquals(5.0 + 8.0, c[CostCategory.CACHE_WRITE], 1e-9)
        assertEquals(20.0, c[CostCategory.OUTPUT], 1e-9)
    }

    @Test
    fun `sonnet read is 0_10, not 0_20`() {
        val c = cost("claude-sonnet-5-5", total = 4_000_000, cached = 1_000_000, write5m = 1_000_000, write1h = 1_000_000, output = 1_000_000)
        assertEquals(2.0, c[CostCategory.REGULAR_INPUT], 1e-9)
        assertEquals(0.10, c[CostCategory.CACHED_INPUT], 1e-9)
        assertEquals(2.50 + 4.0, c[CostCategory.CACHE_WRITE], 1e-9)
        assertEquals(10.0, c[CostCategory.OUTPUT], 1e-9)
    }

    @Test
    fun `haiku at exactly 100K stays on the lower tier`() {
        val c = cost("claude-haiku-5-5", total = 100_000, output = 1_000_000)
        assertEquals(100_000 / 1e6 * 0.10, c[CostCategory.REGULAR_INPUT], 1e-12)
        assertEquals(0.50, c[CostCategory.OUTPUT], 1e-9)
    }

    @Test
    fun `haiku at 100K plus one moves to the upper tier`() {
        val c = cost("claude-haiku-5-5", total = 100_001, output = 1_000_000)
        assertEquals(100_001 / 1e6 * 0.50, c[CostCategory.REGULAR_INPUT], 1e-12)
        assertEquals(2.50, c[CostCategory.OUTPUT], 1e-9)
    }

    @Test
    fun `haiku lower tier writes and read`() {
        val c = cost("claude-haiku-5-5", total = 100_000, cached = 40_000, write5m = 30_000, write1h = 30_000)
        assertEquals(40_000 / 1e6 * 0.01, c[CostCategory.CACHED_INPUT], 1e-12)
        assertEquals(30_000 / 1e6 * 0.125 + 30_000 / 1e6 * 0.20, c[CostCategory.CACHE_WRITE], 1e-12)
        assertEquals(0.0, c[CostCategory.REGULAR_INPUT], 1e-12)
    }

    @Test
    fun `haiku upper tier writes and read`() {
        val c = cost("claude-haiku-5-5", total = 200_000, cached = 100_000, write5m = 50_000, write1h = 50_000)
        assertEquals(100_000 / 1e6 * 0.05, c[CostCategory.CACHED_INPUT], 1e-12)
        assertEquals(50_000 / 1e6 * 0.625 + 50_000 / 1e6 * 1.00, c[CostCategory.CACHE_WRITE], 1e-12)
    }

    @Test
    fun `all three models are registered with limits and resolve by actual name`() {
        assertEquals(3, AnthropicLlms.ALL.size)
        AnthropicLlms.SPECS.forEach {
            assertEquals("anthropic", it.llm.provider)
            assertEquals(1_000_000, it.contextTokens)
            assertEquals(128_000, it.maxOutputTokens)
        }
        assertSame(AnthropicLlms.CLAUDE_SONNET_5_5.llm, AnthropicLlms.findModel("anthropic", "claude-sonnet-5-5"))
        assertFailsWith<IllegalArgumentException> { AnthropicLlms.findModel("anthropic", "claude-unknown") }
        assertFailsWith<IllegalArgumentException> { AnthropicLlms.findModel("google", "claude-opus-5-5") }
    }

    @Test
    fun `minimal thinking form is a constant of the model`() {
        assertSame(AnthropicThinking.BetweenTools, AnthropicLlms.CLAUDE_SONNET_5_5.minimalThinking)
        assertSame(AnthropicThinking.BetweenTools, AnthropicLlms.CLAUDE_OPUS_5_5.minimalThinking)
        assertEquals(AnthropicThinking.Adaptive(OutputConfig.Effort.LOW), AnthropicLlms.CLAUDE_HAIKU_5_5.minimalThinking)
    }

    @Test
    fun `level table maps MINIMAL to the model minimum and the rest to adaptive effort`() {
        val sonnet = AnthropicLlms.CLAUDE_SONNET_5_5
        val haiku = AnthropicLlms.CLAUDE_HAIKU_5_5
        assertSame(AnthropicThinking.BetweenTools, AnthropicThinking.forLevel(sonnet, ThinkingLevel.MINIMAL))
        assertEquals(AnthropicThinking.Adaptive(OutputConfig.Effort.LOW), AnthropicThinking.forLevel(haiku, ThinkingLevel.MINIMAL))
        for (spec in AnthropicLlms.SPECS) {
            assertEquals(OutputConfig.Effort.LOW, AnthropicThinking.forLevel(spec, ThinkingLevel.LOW).effort)
            assertEquals(OutputConfig.Effort.MEDIUM, AnthropicThinking.forLevel(spec, ThinkingLevel.MEDIUM).effort)
            assertEquals(OutputConfig.Effort.HIGH, AnthropicThinking.forLevel(spec, ThinkingLevel.HIGH).effort)
        }
        assertNull(AnthropicThinking.forLevel(sonnet, ThinkingLevel.MINIMAL).effort)
    }

    @Test
    fun `wire params carry the right type`() {
        assertTrue(AnthropicThinking.BetweenTools.thinkingParam().isBetweenTools())
        assertTrue(AnthropicThinking.Adaptive(OutputConfig.Effort.HIGH).thinkingParam().isAdaptive())
    }

    @Test
    fun `every model declares the explicit set of four thinking levels, none is clamped away`() {
        for (spec in AnthropicLlms.SPECS) {
            val modes = spec.llm.capability<Thinking>()!!.modes.filterIsInstance<ThinkingMode.Level>().single()
            assertEquals(ThinkingLevel.entries.toSet(), modes.supported)
            for (level in ThinkingLevel.entries) assertEquals(level, spec.llm.clampThinkingLevel(level))
        }
    }

    @Test
    fun `the provider id of the catalogue is the core constant and only Haiku has a price tier`() {
        assertEquals(LlmProviders.ANTHROPIC, AnthropicLlms.PROVIDER)
        assertTrue(AnthropicLlms.ALL.all { it.provider == LlmProviders.ANTHROPIC })
        assertEquals(listOf(100_000L), AnthropicLlms.CLAUDE_HAIKU_5_5.llm.pricingProfile!!.tiers.map { it.aboveInputTokens })
        assertTrue(AnthropicLlms.CLAUDE_SONNET_5_5.llm.pricingProfile!!.tiers.isEmpty())
        assertTrue(AnthropicLlms.CLAUDE_OPUS_5_5.llm.pricingProfile!!.tiers.isEmpty())
    }
}
