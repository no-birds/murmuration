package org.nobirds.murmuration.gemini

import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Thinking
import org.nobirds.murmuration.llm.ThinkingMode
import org.nobirds.murmuration.llm.capability
import org.nobirds.murmuration.prompt.ThinkingLevel
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * Catalog: the cheap image model `gemini-3.1-flash-lite-image` is declared on both
 * providers with lite pricing and resolves correctly via [GeminiLlms.findModel]
 * (prefix + longest-match does not confuse it with the text `gemini-3.1-flash-lite` or the senior
 * `gemini-3.1-flash-image`).
 */
class GeminiCheapImageModelTest {

    @Test
    fun `vertex cheap image model resolves with lite pricing`() {
        val llm = GeminiLlms.findModel("vertex", "gemini-3.1-flash-lite-image")
        assertEquals("gemini-3.1-flash-lite-image", llm.model)
        val pricing = assertNotNull(llm.pricingProfile)
        assertEquals(0.25, pricing.inputPricePerMillion)
        assertEquals(30.0, pricing.outputPricePerMillion)
        assertEquals(1.5, pricing.thinkingPricePerMillion)
    }

    @Test
    fun `google cheap image model resolves`() {
        val llm = GeminiLlms.findModel("google", "gemini-3.1-flash-lite-image")
        assertEquals("gemini-3.1-flash-lite-image", llm.model)
    }

    @Test
    fun `cheap image model does not shadow siblings`() {
        // The senior image model and the text lite must not be intercepted by the cheap image one.
        assertEquals("gemini-3.1-flash-image", GeminiLlms.findModel("vertex", "gemini-3.1-flash-image").model)
        assertEquals("gemini-3.1-flash-lite", GeminiLlms.findModel("vertex", "gemini-3.1-flash-lite").model)
    }

    @Test
    fun `vertex cheap image model declares MINIMAL and HIGH as the only supported thinking tiers`() {
        // Live probes 2026-07-15: LOW -> 400, MINIMAL and HIGH both work. Without this declaration
        // clampThinkingLevel is a no-op, and the default LOW would take down prod (see LlmCapabilities.kt).
        val llm = GeminiLlms.findModel("vertex", "gemini-3.1-flash-lite-image")
        val supported = llm.capability<Thinking>()?.modes
            ?.filterIsInstance<ThinkingMode.Level>()?.firstOrNull()?.supported
        assertEquals(setOf(ThinkingLevel.MINIMAL, ThinkingLevel.HIGH), supported)
    }

    @Test
    fun `google cheap image model declares MINIMAL and HIGH as the only supported thinking tiers`() {
        val llm = GeminiLlms.findModel("google", "gemini-3.1-flash-lite-image")
        val supported = llm.capability<Thinking>()?.modes
            ?.filterIsInstance<ThinkingMode.Level>()?.firstOrNull()?.supported
        assertEquals(setOf(ThinkingLevel.MINIMAL, ThinkingLevel.HIGH), supported)
    }
}
