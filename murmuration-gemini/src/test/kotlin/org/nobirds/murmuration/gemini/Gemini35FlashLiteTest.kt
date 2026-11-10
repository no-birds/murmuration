package org.nobirds.murmuration.gemini

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * Checks the presence and pricing of Gemini 3.5 Flash-Lite on both providers.
 */
class Gemini35FlashLiteTest {

    @Test
    fun `vertex gemini 3_5 flash lite resolves with correct pricing`() {
        val llm = GeminiLlms.findModel("vertex", "gemini-3.5-flash-lite")
        assertEquals("gemini-3.5-flash-lite", llm.model)
        val pricing = assertNotNull(llm.pricingProfile)
        assertEquals(0.30, pricing.inputPricePerMillion)
        assertEquals(0.03, pricing.cachedInputPricePerMillion)
        assertEquals(2.50, pricing.outputPricePerMillion)
    }

    @Test
    fun `google gemini 3_5 flash lite resolves with correct pricing`() {
        val llm = GeminiLlms.findModel("google", "gemini-3.5-flash-lite")
        assertEquals("gemini-3.5-flash-lite", llm.model)
        val pricing = assertNotNull(llm.pricingProfile)
        assertEquals(0.30, pricing.inputPricePerMillion)
        assertEquals(0.03, pricing.cachedInputPricePerMillion)
        assertEquals(2.50, pricing.outputPricePerMillion)
    }

    @Test
    fun `google gemini models with prefix models slash resolve cleanly`() {
        val lite = GeminiLlms.findModel("google", "models/gemini-3.5-flash-lite")
        assertEquals("gemini-3.5-flash-lite", lite.model)

        val flash37Prefix = GeminiLlms.findModel("google", "models/gemini-3.7-flash")
        assertEquals("gemini-3.7-flash", flash37Prefix.model)

        val flash37NoPrefix = GeminiLlms.findModel("google", "gemini-3.7-flash")
        assertEquals("gemini-3.7-flash", flash37NoPrefix.model)
    }
}
