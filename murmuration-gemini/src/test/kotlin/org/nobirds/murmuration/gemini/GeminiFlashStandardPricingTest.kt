package org.nobirds.murmuration.gemini

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * Gemini 3.6/3.7/3.8 Flash are priced at the standard list rate, not the introductory one:
 * the introduction is a credit back, and a payment made in credits is charged in full.
 */
class GeminiFlashStandardPricingTest {

    @Test
    fun `3_6 3_7 3_8 flash use the standard rate on both providers`() {
        for (provider in listOf("vertex", "google")) {
            for (model in listOf("gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash")) {
                val pricing = assertNotNull(GeminiLlms.findModel(provider, model).pricingProfile)
                assertEquals(1.5, pricing.inputPricePerMillion, "$provider/$model input")
                assertEquals(0.15, pricing.cachedInputPricePerMillion, "$provider/$model cached")
                assertEquals(7.5, pricing.outputPricePerMillion, "$provider/$model output")
            }
        }
    }
}
