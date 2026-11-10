package org.nobirds.murmuration.llm

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class LlmPricingProfileTest {
    @Test
    fun `flat profile has no write prices and tiers`() {
        val flat = LlmPricingProfile(inputPricePerMillion = 1.0, cachedInputPricePerMillion = 0.1, outputPricePerMillion = 2.0)
        assertNull(flat.cacheWriteShortPricePerMillion)
        assertNull(flat.cacheWriteLongPricePerMillion)
        assertTrue(flat.tiers.isEmpty())
    }

    @Test
    fun `withCacheWrite carries both write prices and tiers`() {
        val tier = PriceTier(100_000, PriceRates(6.0, 0.6, 22.5, 7.5, 12.0))
        val p = LlmPricingProfile.withCacheWrite(3.0, 0.3, 15.0, 3.75, 6.0, tiers = listOf(tier))
        assertEquals(3.75, p.cacheWriteShortPricePerMillion)
        assertEquals(6.0, p.cacheWriteLongPricePerMillion)
        assertEquals(listOf(tier), p.tiers)
    }
}
