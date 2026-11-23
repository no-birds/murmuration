package org.nobirds.murmuration.gemini

import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.llm.Thinking
import org.nobirds.murmuration.llm.ThinkingMode
import org.nobirds.murmuration.llm.capability
import org.nobirds.murmuration.prompt.ImageResolution
import org.nobirds.murmuration.prompt.ThinkingLevel
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Catalog: `gemini-nano-banana-2.1` and `gemini-3-pro-image` are declared on both providers with
 * their per-token tariffs, thinking tiers and image resolutions, and are not shadowed by the older
 * `gemini-3.1-flash-image` family in [GeminiLlms.findModel].
 */
class GeminiNanoBananaModelsTest {

    private val providers = listOf("google", "vertex")

    private fun supportedLevels(llm: Llm): Set<ThinkingLevel>? =
        llm.capability<Thinking>()?.modes?.filterIsInstance<ThinkingMode.Level>()?.firstOrNull()?.supported

    @Test
    fun `nano banana 2_1 resolves on both providers with its tariff`() {
        for (provider in providers) {
            val llm = GeminiLlms.findModel(provider, "gemini-nano-banana-2.1")
            assertEquals("gemini-nano-banana-2.1", llm.model)
            assertEquals(provider, llm.provider)
            val pricing = assertNotNull(llm.pricingProfile)
            assertEquals(1.5, pricing.inputPricePerMillion)
            assertEquals(0.15, pricing.cachedInputPricePerMillion)
            assertEquals(30.0, pricing.outputPricePerMillion)
            assertEquals(7.5, pricing.thinkingPricePerMillion)
            assertEquals(0.0, pricing.cacheStoragePricePerMillionPerHour)
        }
    }

    @Test
    fun `nano banana 2_1 declares MINIMAL MEDIUM and HIGH thinking tiers`() {
        for (provider in providers) {
            val llm = GeminiLlms.findModel(provider, "gemini-nano-banana-2.1")
            assertEquals(
                setOf(ThinkingLevel.MINIMAL, ThinkingLevel.MEDIUM, ThinkingLevel.HIGH),
                supportedLevels(llm),
                "provider $provider",
            )
        }
    }

    @Test
    fun `nano banana pro resolves on both providers with its tariff`() {
        for (provider in providers) {
            val llm = GeminiLlms.findModel(provider, "gemini-3-pro-image")
            assertEquals("gemini-3-pro-image", llm.model)
            assertEquals(provider, llm.provider)
            val pricing = assertNotNull(llm.pricingProfile)
            assertEquals(2.0, pricing.inputPricePerMillion)
            assertEquals(0.2, pricing.cachedInputPricePerMillion)
            assertEquals(120.0, pricing.outputPricePerMillion)
            assertEquals(12.0, pricing.thinkingPricePerMillion)
            assertEquals(0.0, pricing.cacheStoragePricePerMillionPerHour)
        }
    }

    @Test
    fun `nano banana pro declares no thinking restriction`() {
        // Thinking tiers of the model were not verified, so none are declared (clamp = pass-through).
        for (provider in providers) {
            assertNull(supportedLevels(GeminiLlms.findModel(provider, "gemini-3-pro-image")), "provider $provider")
        }
    }

    @Test
    fun `new image models do not shadow the older image family`() {
        for (provider in providers) {
            assertEquals("gemini-3.1-flash-image", GeminiLlms.findModel(provider, "gemini-3.1-flash-image").model)
            assertEquals(
                "gemini-3.1-flash-lite-image",
                GeminiLlms.findModel(provider, "gemini-3.1-flash-lite-image").model,
            )
        }
    }

    @Test
    fun `image resolutions never produce the unsupported 512 size for the new models`() {
        // Neither model supports 512; the model-independent mapping clamps SD_512 to 1K.
        val sizes = ImageResolution.entries.map { it.toGemini() }.toSet()
        assertEquals(setOf("1K", "2K", "4K"), sizes)
        assertTrue("512" !in sizes)
    }
}
