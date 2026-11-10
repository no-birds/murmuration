package org.nobirds.murmuration.llm

import org.nobirds.murmuration.llm.CostCategory
import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.prompt.*

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import java.util.UUID
import kotlin.time.Duration.Companion.seconds

class LlmCostCalculatorTest {

    private val flashProfile = LlmPricingProfile(
        inputPricePerMillion = 0.5,
        cachedInputPricePerMillion = 0.05,
        outputPricePerMillion = 3.0,
        cacheStoragePricePerMillionPerHour = 1.0,
    )

    private val imageProfile = LlmPricingProfile(
        inputPricePerMillion = 0.5,
        cachedInputPricePerMillion = 0.0,
        outputPricePerMillion = 60.0,
        cacheStoragePricePerMillionPerHour = 0.0,
    )

    private val flashLlm = Llm(
        provider = "test", family = "gemini", name = "Flash", version = "3", stable = false,
        model = "gemini-3-flash-preview",
        pricingProfile = flashProfile,
    )

    private val imageLlm = Llm(
        provider = "test", family = "imagen", name = "Imagen", version = "3.1", stable = false,
        model = "gemini-3.1-flash-image",
        pricingProfile = imageProfile,
    )

    @BeforeEach
    fun setup() {
        LlmRegistry.clear()
        LlmRegistry.register(listOf(flashLlm, imageLlm))
    }

    @AfterEach
    fun teardown() {
        LlmRegistry.clear()
    }

    private fun rt(total: Long, cached: Long = 0, writeShort: Long = 0, writeLong: Long = 0, output: Long = 0, thinking: Long = 0) =
        RoundTripUsage(InputTokenStatistics(total, cached, writeShort, writeLong), output, thinking)

    @Test
    fun `regular and cached input plus output for Flash`() {
        val cost = LlmCostCalculator.roundTrip(rt(total = 1_000_000, cached = 200_000, output = 500_000), flashProfile)
        assertEquals(0.4, cost[CostCategory.REGULAR_INPUT], 1e-9)
        assertEquals(0.01, cost[CostCategory.CACHED_INPUT], 1e-9)
        assertEquals(1.5, cost[CostCategory.OUTPUT], 1e-9)
        assertEquals(0.0, cost[CostCategory.CREATION], 1e-9)
        assertEquals(0.0, cost[CostCategory.STORAGE], 1e-9)
        assertEquals(0.0, cost[CostCategory.CACHE_WRITE], 1e-9)
        assertEquals(1.91, cost.totalUsd, 1e-9)
    }

    @Test
    fun `image thinking billed at separate cheaper rate, not image-output price`() {
        val imageWithThinking = LlmPricingProfile(
            inputPricePerMillion = 0.5,
            cachedInputPricePerMillion = 0.0,
            outputPricePerMillion = 60.0,
            thinkingPricePerMillion = 3.0,
        )
        // 1M image at $60/1M + 1M thinking at $3/1M, NOT at $60
        val cost = LlmCostCalculator.roundTrip(rt(total = 0, output = 1_000_000, thinking = 1_000_000), imageWithThinking)
        assertEquals(63.0, cost[CostCategory.OUTPUT], 1e-9)
    }

    @Test
    fun `thinking defaults to output price when profile has no thinking rate`() {
        val cost = LlmCostCalculator.roundTrip(rt(total = 0, output = 100_000, thinking = 100_000), flashProfile)
        // no thinkingPrice → both at $3: (100k + 100k) × 3 / 1M = 0.6
        assertEquals(0.6, cost[CostCategory.OUTPUT], 1e-9)
    }

    @Test
    fun `cache object is billed at full input price plus storage`() {
        val tokens = 1_000_000L
        val cost = LlmCostCalculator.cacheObject(CacheObjectUsage(tokens, tokenHours = tokens.toDouble() * 1.0), flashProfile)
        assertEquals(0.5, cost[CostCategory.CREATION], 1e-9)
        assertEquals(1.0, cost[CostCategory.STORAGE], 1e-9)
        assertEquals(1.5, cost.totalUsd, 1e-9)
    }

    @Test
    fun `null profile yields zero cost`() {
        assertEquals(LlmCostBreakdown.ZERO, LlmCostCalculator.roundTrip(rt(total = 9_999_999, output = 9_999_999), profile = null))
        assertEquals(LlmCostBreakdown.ZERO, LlmCostCalculator.cacheObject(CacheObjectUsage(9_999_999, 1.0), profile = null))
    }

    // --- S6b: cost is per RT, there is no aggregate on input ---

    private val haikuProfile = LlmPricingProfile.withCacheWrite(
        inputPricePerMillion = 1.0,
        cachedInputPricePerMillion = 0.1,
        outputPricePerMillion = 5.0,
        cacheWriteShortPricePerMillion = 1.25,
        cacheWriteLongPricePerMillion = 2.0,
        tiers = listOf(
            PriceTier(
                aboveInputTokens = 100_000,
                rates = PriceRates(
                    inputPricePerMillion = 2.0,
                    cachedInputPricePerMillion = 0.2,
                    outputPricePerMillion = 10.0,
                    cacheWriteShortPricePerMillion = 2.5,
                    cacheWriteLongPricePerMillion = 4.0,
                ),
            ),
        ),
    )

    @Test
    fun `tier is chosen by the full input of one round-trip, 100000 inclusive in the lower tier`() {
        val low = LlmCostCalculator.roundTrip(rt(total = 100_000), haikuProfile)
        val high = LlmCostCalculator.roundTrip(rt(total = 100_001), haikuProfile)
        assertEquals(0.1, low[CostCategory.REGULAR_INPUT], 1e-9)
        assertEquals(100_001 / 1e6 * 2.0, high[CostCategory.REGULAR_INPUT], 1e-9)
    }

    @Test
    fun `tier threshold is by full input including cached and writes`() {
        // 60K cached + 50K write5m = 110K full input → upper tier, although regular = 0
        val cost = LlmCostCalculator.roundTrip(rt(total = 110_000, cached = 60_000, writeShort = 50_000), haikuProfile)
        assertEquals(0.0, cost[CostCategory.REGULAR_INPUT], 1e-9)
        assertEquals(60_000 / 1e6 * 0.2, cost[CostCategory.CACHED_INPUT], 1e-9)
        assertEquals(50_000 / 1e6 * 2.5, cost[CostCategory.CACHE_WRITE], 1e-9)
    }

    @Test
    fun `Haiku 90K and 110K round-trips are priced by their own tiers and the aggregate differs from the sum calculation`() {
        val a = LlmCostCalculator.roundTrip(rt(total = 90_000, output = 1_000), haikuProfile)
        val b = LlmCostCalculator.roundTrip(rt(total = 110_000, output = 1_000), haikuProfile)
        assertEquals(90_000 / 1e6 * 1.0 + 1_000 / 1e6 * 5.0, a.totalUsd, 1e-9)
        assertEquals(110_000 / 1e6 * 2.0 + 1_000 / 1e6 * 10.0, b.totalUsd, 1e-9)
        val aggregate = a + b
        // "cost of the token sum" — RT 200K at the upper tier — would have been a different number
        val ofSums = LlmCostCalculator.roundTrip(rt(total = 200_000, output = 2_000), haikuProfile)
        assertTrue(aggregate.totalUsd < ofSums.totalUsd, "aggregate=${aggregate.totalUsd} ofSums=${ofSums.totalUsd}")
    }

    @Test
    fun `cache write is billed at write prices by TTL class, not as regular input`() {
        val cost = LlmCostCalculator.roundTrip(rt(total = 30_000, cached = 5_000, writeShort = 10_000, writeLong = 4_000), haikuProfile)
        assertEquals(11_000 / 1e6 * 1.0, cost[CostCategory.REGULAR_INPUT], 1e-9)
        assertEquals(5_000 / 1e6 * 0.1, cost[CostCategory.CACHED_INPUT], 1e-9)
        assertEquals(10_000 / 1e6 * 1.25 + 4_000 / 1e6 * 2.0, cost[CostCategory.CACHE_WRITE], 1e-9)
        assertEquals(0.0, cost[CostCategory.CREATION], 1e-9)
        assertEquals(cost[CostCategory.REGULAR_INPUT] + cost[CostCategory.CACHED_INPUT] + cost[CostCategory.CACHE_WRITE], cost.totalUsd, 1e-12)
    }

    @Test
    fun `flat profile with write prices (Sonnet shape) bills writes at write prices, not as input`() {
        val sonnet = LlmPricingProfile.withCacheWrite(
            inputPricePerMillion = 2.0,
            cachedInputPricePerMillion = 0.10,
            outputPricePerMillion = 10.0,
            cacheWriteShortPricePerMillion = 2.50,
            cacheWriteLongPricePerMillion = 4.0,
        )
        val cost = LlmCostCalculator.roundTrip(rt(total = 300_000, cached = 100_000, writeShort = 50_000, writeLong = 20_000, output = 1_000), sonnet)
        assertEquals(130_000 / 1e6 * 2.0, cost[CostCategory.REGULAR_INPUT], 1e-9)
        assertEquals(100_000 / 1e6 * 0.10, cost[CostCategory.CACHED_INPUT], 1e-9)
        assertEquals(50_000 / 1e6 * 2.50 + 20_000 / 1e6 * 4.0, cost[CostCategory.CACHE_WRITE], 1e-9)
        assertEquals(1_000 / 1e6 * 10.0, cost[CostCategory.OUTPUT], 1e-9)
    }

    @Test
    fun `profile without write prices bills a write as regular input`() {
        val cost = LlmCostCalculator.roundTrip(rt(total = 10_000, writeShort = 4_000), flashProfile)
        assertEquals(4_000 / 1e6 * 0.5, cost[CostCategory.CACHE_WRITE], 1e-9)
        assertEquals(10_000 / 1e6 * 0.5, cost[CostCategory.REGULAR_INPUT] + cost[CostCategory.CACHE_WRITE], 1e-9)
    }

    // --- S6b: Gemini/Azure numbers do not change ---

    /** The former "cost of the token sum" formula — to verify that the fold over RT gives the same numbers. */
    private fun legacyCost(rows: List<Triple<Long, Long, Long>>, thinking: Long, profile: LlmPricingProfile): Double {
        val inputTotal = rows.sumOf { it.first }
        val inputCached = rows.sumOf { it.second }
        val output = rows.sumOf { it.third }
        val regular = (inputTotal - inputCached).coerceAtLeast(0).toDouble()
        return regular / 1e6 * profile.inputPricePerMillion +
            inputCached.toDouble() / 1e6 * profile.cachedInputPricePerMillion +
            output.toDouble() / 1e6 * profile.outputPricePerMillion +
            thinking.toDouble() / 1e6 * profile.effectiveThinkingPricePerMillion
    }

    @Test
    fun `Gemini and Azure rows - fold by round-trip equals the old formula on the sums, including cache object`() {
        val azureProfile = LlmPricingProfile(inputPricePerMillion = 2.0, cachedInputPricePerMillion = 0.5, outputPricePerMillion = 8.0)
        for (profile in listOf(flashProfile, azureProfile)) {
            val rows = listOf(Triple(120_000L, 100_000L, 800L), Triple(130_000L, 110_000L, 1_200L), Triple(5_000L, 0L, 50L))
            val thinking = 400L
            val fold = rows.fold(LlmCostBreakdown.ZERO) { acc, (total, cached, out) ->
                acc + LlmCostCalculator.roundTrip(rt(total, cached, output = out, thinking = thinking / 3), profile)
            } + LlmCostCalculator.cacheObject(CacheObjectUsage(110_000, 110_000 * 0.25), profile)
            // thinking is split by rows: 3 × (400/3 = 133) — we reconcile against the same numbers
            val expected = legacyCost(rows, 3 * (thinking / 3), profile) +
                110_000 / 1e6 * profile.inputPricePerMillion +
                110_000 * 0.25 / 1e6 * profile.cacheStoragePricePerMillionPerHour
            assertEquals(expected, fold.totalUsd, 1e-9)
            assertEquals(0.0, fold[CostCategory.CACHE_WRITE], 0.0)
        }
    }

    @Test
    fun `row with cached greater than prompt has regular zeroed, not negative (accepted deviation R-F2-2)`() {
        val good = LlmCostCalculator.roundTrip(rt(total = 10_000, cached = 4_000), flashProfile)
        val odd = LlmCostCalculator.roundTrip(rt(total = 1_000, cached = 3_000), flashProfile)
        assertEquals(0.0, odd[CostCategory.REGULAR_INPUT], 0.0)
        assertEquals(3_000 / 1e6 * 0.05, odd[CostCategory.CACHED_INPUT], 1e-12)
        // the old formula on the sum would have subtracted a negative regular (−2000) from the regular of the good row
        val oldSumRegular = (11_000L - 7_000L) / 1e6 * 0.5
        val newSumRegular = (good + odd)[CostCategory.REGULAR_INPUT]
        assertEquals(6_000 / 1e6 * 0.5, newSumRegular, 1e-12)
        assertTrue(newSumRegular > oldSumRegular)
    }

    @Test
    fun `perModel resolves profile via LlmRegistry by actual model name with version suffix`() = runBlocking {
        val collector = PipelineStatisticsCollector()
        val pipelineId = UUID.randomUUID()

        // The API name contains a date suffix that the API appends to the requested alias.
        val flashEntry = PromptExecutionEntry(
            "FlashAgent",
            1.seconds,
            PromptExecutionStatistics(
                modelName = "gemini-3-flash-preview-04-17",
                requestedModelName = "gemini-3-flash-preview",
                tokens = PromptExecutionTokenStatistics(
                    totalTokens = 1500,
                    input = InputTokenStatistics(total = 1000, cached = 200, cacheWriteShort = 0, cacheWriteLong = 0),
                    output = OutputTokenStatistics(total = 500),
                ),
            ),
        )
        val imageEntry = PromptExecutionEntry(
            "ImageAgent",
            2.seconds,
            PromptExecutionStatistics(
                modelName = "gemini-3.1-flash-image-09-01",
                requestedModelName = "gemini-3.1-flash-image",
                tokens = PromptExecutionTokenStatistics(
                    totalTokens = 1290,
                    input = InputTokenStatistics(total = 100, cached = 0, cacheWriteShort = 0, cacheWriteLong = 0),
                    output = OutputTokenStatistics(total = 1290),
                ),
            ),
        )
        collector.onEntryReported(pipelineId, flashEntry)
        collector.onEntryReported(pipelineId, imageEntry)

        val byModel = LlmCostCalculator.perModel(collector.statistics)
        assertEquals(2, byModel.size)
        // Flash text output = 500 * $3 / 1M = 0.0015
        val flashCost = byModel["gemini-3-flash-preview-04-17"]!!
        assertEquals(0.0015, flashCost[CostCategory.OUTPUT], 1e-9)
    }

    @Test
    fun `perModel returns null entry for unknown model`() = runBlocking {
        val collector = PipelineStatisticsCollector()
        val pipelineId = UUID.randomUUID()

        collector.onEntryReported(
            pipelineId,
            PromptExecutionEntry(
                "UnknownAgent",
                1.seconds,
                PromptExecutionStatistics(
                    modelName = "completely-unknown-model",
                    requestedModelName = "completely-unknown-model",
                    tokens = PromptExecutionTokenStatistics(
                        input = InputTokenStatistics(total = 1000, cached = 0, cacheWriteShort = 0, cacheWriteLong = 0),
                        output = OutputTokenStatistics(total = 500),
                    ),
                ),
            ),
        )

        val byModel = LlmCostCalculator.perModel(collector.statistics)
        assertEquals(1, byModel.size)
        assertNull(byModel["completely-unknown-model"], "Unknown model must yield null in byModel")
    }

    @Test
    fun `perModel uses cache info from prompt entries for the right model`() = runBlocking {
        val collector = PipelineStatisticsCollector()
        val pipelineId = UUID.randomUUID()

        val ttlMillis = 30L * 60_000L
        val cacheTokens = 500_000L
        val entry = PromptExecutionEntry(
            "CachingAgent",
            1.seconds,
            PromptExecutionStatistics(
                modelName = "gemini-3-flash-preview-04-17",
                requestedModelName = "gemini-3-flash-preview",
                tokens = PromptExecutionTokenStatistics(
                    totalTokens = 500_000,
                    input = InputTokenStatistics(total = cacheTokens, cached = 0, cacheWriteShort = 0, cacheWriteLong = 0),
                    output = OutputTokenStatistics(total = 0),
                ),
                cacheId = "cache-1",
                cachedTokenCount = cacheTokens,
                isCacheCreated = true,
                cacheTtlMillis = ttlMillis,
            ),
        )
        collector.onEntryReported(pipelineId, entry)

        val cost = LlmCostCalculator.perModel(collector.statistics)["gemini-3-flash-preview-04-17"]!!
        assertEquals(0.25, cost[CostCategory.CREATION], 1e-9)
        assertEquals(0.25, cost[CostCategory.STORAGE], 1e-9)
    }

    @Test
    fun `perModel computes creation from the CACHE_CREATE entry (size in input total)`() = runBlocking {
        // Regression: the cache size lives in the CACHE_CREATE entry (tokens.input.total), the isCacheCreated flag
        // is on the same one, and cachedTokenCount=0. The old code filtered only ROUND_TRIP → creation=$0
        // (a bug in the Telegram report: "1 created (110879 tok) — creation=$0").
        val collector = PipelineStatisticsCollector()
        val pipelineId = UUID.randomUUID()
        val ttlMillis = 30L * 60_000L
        val cacheTokens = 500_000L
        val cacheCreate = PromptExecutionEntry(
            "CachingAgent",
            1.seconds,
            PromptExecutionStatistics(
                modelName = "gemini-3-flash-preview-04-17",
                requestedModelName = "gemini-3-flash-preview",
                tokens = PromptExecutionTokenStatistics(
                    totalTokens = cacheTokens,
                    input = InputTokenStatistics(total = cacheTokens, cached = 0, cacheWriteShort = 0, cacheWriteLong = 0),
                    output = OutputTokenStatistics(total = 0),
                ),
                cacheId = "cache-1",
                cachedTokenCount = 0,
                isCacheCreated = true,
                cacheTtlMillis = ttlMillis,
            ),
            ExecutionNature.CACHE_CREATE,
        )
        collector.onEntryReported(pipelineId, cacheCreate)

        val cost = LlmCostCalculator.perModel(collector.statistics)["gemini-3-flash-preview-04-17"]!!
        // 500K × $0.5/M = $0.25 creation; storage 500K × 0.5h × $1/M/h = $0.25.
        assertEquals(0.25, cost[CostCategory.CREATION], 1e-9)
        assertEquals(0.25, cost[CostCategory.STORAGE], 1e-9)
    }

    @Test
    fun `cache storage uses the actual lifetime from onCacheDeleted, not the declared TTL`() = runBlocking {
        // Defect regression: storage was counted by the declared TTL (upper bound), although a PIPELINE-scope
        // cache actually lives until the end of the pipeline — shorter than the TTL by orders of magnitude (in prod: 22s vs 10m).
        val collector = PipelineStatisticsCollector()
        val pipelineId = UUID.randomUUID()

        val declaredTtlMillis = 10L * 60_000L // declared TTL: 10 minutes.
        val actualLifetimeMillis = 22_000L // the cache really lived 22 seconds (the pipeline finished quickly).
        val cacheTokens = 100_000L
        val cacheCreate = PromptExecutionEntry(
            "CachingAgent",
            1.seconds,
            PromptExecutionStatistics(
                modelName = "gemini-3-flash-preview-04-17",
                requestedModelName = "gemini-3-flash-preview",
                tokens = PromptExecutionTokenStatistics(
                    totalTokens = cacheTokens,
                    input = InputTokenStatistics(total = cacheTokens, cached = 0, cacheWriteShort = 0, cacheWriteLong = 0),
                    output = OutputTokenStatistics(total = 0),
                ),
                cacheId = "cache-real-lifetime",
                cachedTokenCount = 0,
                isCacheCreated = true,
                cacheTtlMillis = declaredTtlMillis,
            ),
            ExecutionNature.CACHE_CREATE,
        )
        collector.onEntryReported(pipelineId, cacheCreate)
        collector.onCacheDeleted(pipelineId, "cache-real-lifetime", actualLifetimeMillis)

        val stats = collector.statistics
        assertEquals(actualLifetimeMillis, stats.cacheLifetimeMillis["cache-real-lifetime"])

        val cost = LlmCostCalculator.perModel(stats)["gemini-3-flash-preview-04-17"]!!
        val expectedStorage = cacheTokens.toDouble() * (actualLifetimeMillis / 3_600_000.0) / 1_000_000.0 *
            flashProfile.cacheStoragePricePerMillionPerHour
        val ttlBasedStorage = cacheTokens.toDouble() * (declaredTtlMillis / 3_600_000.0) / 1_000_000.0 *
            flashProfile.cacheStoragePricePerMillionPerHour
        assertEquals(expectedStorage, cost[CostCategory.STORAGE], 1e-9)
        // Catch a regression if the fix silently stopped working: the TTL-based number is noticeably larger.
        assertTrue(expectedStorage < ttlBasedStorage, "expected $expectedStorage < ttl-based $ttlBasedStorage")
    }

    @Test
    fun `cache storage falls back to the declared TTL until onCacheDeleted arrives`() = runBlocking {
        // Typical case for the "Pipeline Finished" report: cleanup is asynchronous and races with reading
        // statistics right after the pipeline — onCacheDeleted often has not arrived by then.
        val collector = PipelineStatisticsCollector()
        val pipelineId = UUID.randomUUID()

        val ttlMillis = 30L * 60_000L
        val cacheTokens = 500_000L
        val cacheCreate = PromptExecutionEntry(
            "CachingAgent",
            1.seconds,
            PromptExecutionStatistics(
                modelName = "gemini-3-flash-preview-04-17",
                requestedModelName = "gemini-3-flash-preview",
                tokens = PromptExecutionTokenStatistics(
                    totalTokens = cacheTokens,
                    input = InputTokenStatistics(total = cacheTokens, cached = 0, cacheWriteShort = 0, cacheWriteLong = 0),
                    output = OutputTokenStatistics(total = 0),
                ),
                cacheId = "cache-not-deleted-yet",
                cachedTokenCount = 0,
                isCacheCreated = true,
                cacheTtlMillis = ttlMillis,
            ),
            ExecutionNature.CACHE_CREATE,
        )
        collector.onEntryReported(pipelineId, cacheCreate)
        // we do NOT call onCacheDeleted.

        val stats = collector.statistics
        assertTrue(stats.cacheLifetimeMillis.isEmpty())

        val cost = LlmCostCalculator.perModel(stats)["gemini-3-flash-preview-04-17"]!!
        // Fallback to the declared TTL (0.5h): 500K × 0.5h × $1/M/h = $0.25.
        assertEquals(0.25, cost[CostCategory.STORAGE], 1e-9)
    }

    @Test
    fun `cache storage uses the elapsed estimate for PIPELINE scope until the exact time arrives`() = runBlocking {
        // Design defect C2: even with onCacheDeleted forwarded, the actual lifetime is almost never
        // known by the time of the report (cache deletion is async fire-and-forget, racing with reading
        // statistics). But a PIPELINE-scope cache dies together with the pipeline, so at finish time
        // its lifetime ≈ (finish − creation) — orders of magnitude less than the declared TTL. We check that
        // storage is counted by the elapsed estimate, not by TTL (otherwise cost is overstated many times over).
        val collector = PipelineStatisticsCollector()
        val pipelineId = UUID.randomUUID()

        val declaredTtlMillis = 10L * 60_000L // TTL: 10 minutes.
        val cacheTokens = 100_000L
        val cacheCreate = PromptExecutionEntry(
            "CachingAgent",
            1.seconds,
            PromptExecutionStatistics(
                modelName = "gemini-3-flash-preview-04-17",
                requestedModelName = "gemini-3-flash-preview",
                tokens = PromptExecutionTokenStatistics(
                    totalTokens = cacheTokens,
                    input = InputTokenStatistics(total = cacheTokens, cached = 0, cacheWriteShort = 0, cacheWriteLong = 0),
                    output = OutputTokenStatistics(total = 0),
                ),
                cacheId = "cache-pipeline-scoped",
                cachedTokenCount = 0,
                isCacheCreated = true,
                cacheTtlMillis = declaredTtlMillis,
                cacheScope = PromptCacheScope.PIPELINE,
            ),
            ExecutionNature.CACHE_CREATE,
        )
        collector.onEntryReported(pipelineId, cacheCreate)
        // onCacheDeleted did NOT arrive (a typical race). The pipeline finished — finish stamp:
        collector.onFinished(pipelineId)

        val stats = collector.statistics
        // There is no exact lifetime, but there is an elapsed estimate — and it is much less than the TTL.
        assertTrue(stats.cacheLifetimeMillis.isEmpty(), "exact map must be empty")
        val estimate = stats.cacheEstimatedLifetimeMillis["cache-pipeline-scoped"]
        assertTrue(estimate != null && estimate >= 0 && estimate < declaredTtlMillis,
            "elapsed estimate $estimate must be in [0, $declaredTtlMillis)")

        val cost = LlmCostCalculator.perModel(stats)["gemini-3-flash-preview-04-17"]!!
        val ttlBasedStorage = cacheTokens.toDouble() * (declaredTtlMillis / 3_600_000.0) / 1_000_000.0 *
            flashProfile.cacheStoragePricePerMillionPerHour
        val expectedStorage = cacheTokens.toDouble() * (estimate!! / 3_600_000.0) / 1_000_000.0 *
            flashProfile.cacheStoragePricePerMillionPerHour
        assertEquals(expectedStorage, cost[CostCategory.STORAGE], 1e-9)
        // Regression gate: if the estimate silently stopped working and the TTL fallback remained — the number would be larger.
        assertTrue(cost[CostCategory.STORAGE] < ttlBasedStorage,
            "storage ${cost[CostCategory.STORAGE]} must be less than TTL-based $ttlBasedStorage")
    }

    @Test
    fun `elapsed estimate does NOT apply to non-PIPELINE scope — the TTL fallback remains`() = runBlocking {
        // A TTL_BASED cache outlives the pipeline (CachePipelineCleaner does not clean it), so elapsed from
        // finish would be a lie — an honest fallback to TTL as the upper bound.
        val collector = PipelineStatisticsCollector()
        val pipelineId = UUID.randomUUID()
        val ttlMillis = 30L * 60_000L
        val cacheTokens = 500_000L
        val cacheCreate = PromptExecutionEntry(
            "CachingAgent",
            1.seconds,
            PromptExecutionStatistics(
                modelName = "gemini-3-flash-preview-04-17",
                requestedModelName = "gemini-3-flash-preview",
                tokens = PromptExecutionTokenStatistics(
                    totalTokens = cacheTokens,
                    input = InputTokenStatistics(total = cacheTokens, cached = 0, cacheWriteShort = 0, cacheWriteLong = 0),
                    output = OutputTokenStatistics(total = 0),
                ),
                cacheId = "cache-ttl-based",
                cachedTokenCount = 0,
                isCacheCreated = true,
                cacheTtlMillis = ttlMillis,
                cacheScope = PromptCacheScope.TTL_BASED,
            ),
            ExecutionNature.CACHE_CREATE,
        )
        collector.onEntryReported(pipelineId, cacheCreate)
        collector.onFinished(pipelineId)

        val stats = collector.statistics
        assertTrue(stats.cacheEstimatedLifetimeMillis.isEmpty(), "non-PIPELINE scope must not get an estimate")
        val cost = LlmCostCalculator.perModel(stats)["gemini-3-flash-preview-04-17"]!!
        // Fallback to TTL (0.5h): 500K × 0.5h × $1/M/h = $0.25.
        assertEquals(0.25, cost[CostCategory.STORAGE], 1e-9)
    }

    @Test
    fun `findByActualModel picks longest matching model on collisions`() {
        val short = Llm(
            provider = "test", family = "f", name = "short", version = "1", stable = false,
            model = "gemini-3-flash",
            pricingProfile = flashProfile,
        )
        val long = Llm(
            provider = "test", family = "f", name = "long", version = "1", stable = false,
            model = "gemini-3-flash-preview",
            pricingProfile = imageProfile,
        )
        LlmRegistry.clear()
        LlmRegistry.register(listOf(short, long))

        val resolved = LlmRegistry.findByActualModel("gemini-3-flash-preview-04-17")
        assertEquals(long, resolved)
    }
}
