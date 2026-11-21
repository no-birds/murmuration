package org.nobirds.murmuration.gemini

import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Caching
import org.nobirds.murmuration.llm.Thinking
import org.nobirds.murmuration.llm.ThinkingMode
import org.nobirds.murmuration.llm.ToolCalling
import org.nobirds.murmuration.llm.capability
import org.nobirds.murmuration.llm.clampThinkingLevel
import org.nobirds.murmuration.prompt.ThinkingLevel
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull

/**
 * Checks the Gemini 4 line (Hydrogen, Carbon, Argon) on both providers: resolution by id,
 * pricing, declared thinking tiers and capabilities. Neon is not in the catalog.
 */
class Gemini4ModelsTest {

    private data class Expected(val model: String, val input: Double, val cached: Double, val output: Double)

    private val expected = listOf(
        Expected("gemini-4-hydrogen", 1.5, 0.075, 7.5),
        Expected("gemini-4-carbon", 4.0, 0.2, 20.0),
        Expected("gemini-4-argon", 4.0, 0.2, 20.0),
    )

    private val providers = listOf("vertex", "google")

    @Test
    fun `gemini 4 models resolve with global standard pricing on both providers`() {
        for (provider in providers) for (e in expected) {
            val llm = GeminiLlms.findModel(provider, e.model)
            assertEquals(e.model, llm.model)
            assertEquals(provider, llm.provider)
            val pricing = assertNotNull(llm.pricingProfile)
            assertEquals(e.input, pricing.inputPricePerMillion, "$provider/${e.model} input")
            assertEquals(e.cached, pricing.cachedInputPricePerMillion, "$provider/${e.model} cached")
            assertEquals(e.output, pricing.outputPricePerMillion, "$provider/${e.model} output")
            // No explicit cache on Gemini 4: nothing to bill for storage.
            assertEquals(0.0, pricing.cacheStoragePricePerMillionPerHour, "$provider/${e.model} storage")
        }
    }

    @Test
    fun `gemini 4 models declare LOW MEDIUM HIGH and clamp MINIMAL up to LOW`() {
        val tiers = setOf(ThinkingLevel.LOW, ThinkingLevel.MEDIUM, ThinkingLevel.HIGH)
        for (provider in providers) for (e in expected) {
            val llm = GeminiLlms.findModel(provider, e.model)
            val supported = llm.capability<Thinking>()?.modes
                ?.filterIsInstance<ThinkingMode.Level>()?.firstOrNull()?.supported
            assertEquals(tiers, supported, "$provider/${e.model}")
            // MINIMAL is a 400 on Gemini 4; the declaration lets callers clamp instead of crashing.
            assertEquals(ThinkingLevel.LOW, llm.clampThinkingLevel(ThinkingLevel.MINIMAL))
            assertEquals(ThinkingLevel.HIGH, llm.clampThinkingLevel(ThinkingLevel.HIGH))
        }
    }

    @Test
    fun `gemini 4 models are text models with implicit cache floor and tool calling`() {
        for (provider in providers) for (e in expected) {
            val llm = GeminiLlms.findModel(provider, e.model)
            assertEquals("gemini", llm.family)
            assertEquals(4096, llm.capability<Caching>()?.minTokens)
            // Gemini 4 has no explicit context cache: the cache manager must not call cache.create.
            assertEquals(false, llm.capability<Caching>()?.explicit, "$provider/${e.model}")
            assertNotNull(llm.capability<ToolCalling>())
        }
    }

    @Test
    fun `older gemini text models keep explicit caching`() {
        for (provider in providers) {
            assertEquals(true, GeminiLlms.findModel(provider, "gemini-3.8-flash").capability<Caching>()?.explicit)
        }
    }

    @Test
    fun `models slash prefix resolves on the direct api`() {
        assertEquals("gemini-4-hydrogen", GeminiLlms.findModel("google", "models/gemini-4-hydrogen").model)
    }

    @Test
    fun `gemini 4 neon is not in the catalog`() {
        assertFailsWith<IllegalArgumentException> { GeminiLlms.findModel("vertex", "gemini-4-neon") }
    }
}
