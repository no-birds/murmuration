package org.nobirds.murmuration.llm

import org.nobirds.murmuration.prompt.InputTokenStatistics
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** The categories live only in [CostCategory]: the breakdown has no per-category fields to forget. */
class LlmCostBreakdownTest {

    @Test
    fun `every category enters totalUsd and plus without a code change`() {
        CostCategory.entries.forEach { c ->
            val only = LlmCostBreakdown.of { if (it == c) 2.0 else 0.0 }
            assertEquals(2.0, only.totalUsd, 0.0, "$c is missing from totalUsd")
            val doubled = only + only
            assertEquals(4.0, doubled[c], 0.0, "$c is missing from plus")
            assertEquals(4.0, doubled.totalUsd, 0.0)
        }
    }

    @Test
    fun `no named per-category usd property exists - only the derived total`() {
        val named = LlmCostBreakdown::class.java.methods.map { it.name }.filter { it.startsWith("get") && it.endsWith("Usd") }
        assertEquals(listOf("getTotalUsd"), named)
    }

    @Test
    fun `factories fill only the categories of their subject`() {
        val profile = LlmPricingProfile(inputPricePerMillion = 1.0, cachedInputPricePerMillion = 0.1, outputPricePerMillion = 2.0, perRequestUsd = 0.5)
        val rt = LlmCostCalculator.roundTrip(RoundTripUsage(InputTokenStatistics(1_000_000, 0, 0, 0), output = 1_000_000), profile)
        CostCategory.entries.filter { it.subject == CostSubject.CACHE_OBJECT }.forEach { assertEquals(0.0, rt[it]) }
        val obj = LlmCostCalculator.cacheObject(CacheObjectUsage(1_000_000, tokenHours = 1_000_000.0), profile)
        CostCategory.entries.filter { it.subject == CostSubject.ROUND_TRIP }.forEach { assertEquals(0.0, obj[it]) }
        assertTrue(rt.totalUsd > 0.0 && obj[CostCategory.CREATION] > 0.0)
    }

    @Test
    fun `subject factory rejects a foreign or missing category`() {
        assertFailsWith<IllegalArgumentException> {
            LlmCostBreakdown.of(CostSubject.CACHE_OBJECT, mapOf(CostCategory.CREATION to 1.0))
        }
        assertFailsWith<IllegalArgumentException> {
            LlmCostBreakdown.of(CostSubject.CACHE_OBJECT, mapOf(CostCategory.CREATION to 1.0, CostCategory.STORAGE to 1.0, CostCategory.OUTPUT to 1.0))
        }
    }
}
