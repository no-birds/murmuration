package org.nobirds.murmuration.llm

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.prompt.*
import java.util.EnumMap

/**
 * Cost in USD as a total function "[CostCategory] -> USD". There are no named per-category fields: the
 * categories live only in [CostCategory], so [totalUsd], [plus] and [get] walk [CostCategory.entries] and a
 * new category is priced, summed and added up without touching this class. Every instance holds ALL
 * categories (constructor is private; [of] asks the caller for each one).
 */
class LlmCostBreakdown private constructor(private val amounts: Map<CostCategory, Double>) {

    val totalUsd: Double
        get() = CostCategory.entries.sumOf { amounts.getValue(it) }

    /** USD of one [category]. */
    operator fun get(category: CostCategory): Double = amounts.getValue(category)

    operator fun plus(other: LlmCostBreakdown) = of { this[it] + other[it] }

    override fun equals(other: Any?) = other is LlmCostBreakdown && amounts == other.amounts

    override fun hashCode() = amounts.hashCode()

    override fun toString() = CostCategory.entries.joinToString(prefix = "LlmCostBreakdown(", postfix = ")") { "$it=${this[it]}" }

    companion object {
        val ZERO = of { 0.0 }

        /** Builds a breakdown from a total function over [CostCategory]; every category is named by the caller. */
        fun of(amount: (CostCategory) -> Double) =
            LlmCostBreakdown(CostCategory.entries.associateWithTo(EnumMap(CostCategory::class.java), amount))

        /**
         * Builds the breakdown of one [subject]: [amounts] must name exactly the categories of that subject
         * (a round-trip knows nothing of cache-object categories and vice versa); the rest are zero.
         * Not expressible by type — the key set depends on the enum — hence the `require`.
         */
        fun of(subject: CostSubject, amounts: Map<CostCategory, Double>): LlmCostBreakdown {
            val own = CostCategory.entries.filter { it.subject == subject }.toSet()
            require(amounts.keys == own) { "Categories of $subject must be exactly $own, got ${amounts.keys}" }
            return of { amounts[it] ?: 0.0 }
        }
    }
}

/**
 * Usage of ONE round-trip — the only token input of the calculator. The price tier is chosen by the
 * full input of THIS round-trip, so an aggregate of several round-trips cannot be expressed: the
 * cost of an aggregate is the sum of [LlmCostCalculator.roundTrip] costs.
 */
data class RoundTripUsage(
    val input: InputTokenStatistics,
    /** Completion tokens, billed at the output price. */
    val output: Long = 0,
    /** Thinking tokens, billed at the thinking price (for text models = output price). */
    val thinking: Long = 0,
)

/**
 * Usage of one Gemini cache OBJECT: created tokens (billed once at the input price) and the storage
 * basis [tokenHours] = created tokens × lifetime in hours (lifetime precedence: see
 * `resolveCacheLifetimeMillis`).
 */
data class CacheObjectUsage(
    val createdTokens: Long,
    val tokenHours: Double = 0.0,
)

object LlmCostCalculator {

    private const val PER_MILLION = 1_000_000.0
    private const val MILLIS_PER_HOUR = 3_600_000.0

    /**
     * Cost of ONE round-trip. Returns [LlmCostBreakdown.ZERO] when no profile is given. The rates are
     * those of the tier whose threshold is exceeded by the full input of this round-trip (or the base
     * rates); regular input = total − cached − cache writes (never negative). Writes are billed at the
     * write prices of the profile; a flat profile has none, and a write on it is billed as regular
     * input rather than dropped.
     */
    fun roundTrip(usage: RoundTripUsage, profile: LlmPricingProfile?): LlmCostBreakdown {
        if (profile == null) return LlmCostBreakdown.ZERO
        val input = usage.input
        val tier = profile.tiers.filter { input.total > it.aboveInputTokens }.maxByOrNull { it.aboveInputTokens }
        val inputPrice = tier?.rates?.inputPricePerMillion ?: profile.inputPricePerMillion
        val cachedPrice = tier?.rates?.cachedInputPricePerMillion ?: profile.cachedInputPricePerMillion
        val outputPrice = tier?.rates?.outputPricePerMillion ?: profile.outputPricePerMillion
        val thinkingPrice = if (tier != null) outputPrice else profile.effectiveThinkingPricePerMillion
        val writeShortPrice = tier?.rates?.cacheWriteShortPricePerMillion ?: profile.cacheWriteShortPricePerMillion ?: inputPrice
        val writeLongPrice = tier?.rates?.cacheWriteLongPricePerMillion ?: profile.cacheWriteLongPricePerMillion ?: inputPrice
        return LlmCostBreakdown.of(
            CostSubject.ROUND_TRIP,
            mapOf(
                CostCategory.REGULAR_INPUT to input.uncached.toDouble() / PER_MILLION * inputPrice,
                CostCategory.CACHED_INPUT to input.cached.toDouble() / PER_MILLION * cachedPrice,
                CostCategory.CACHE_WRITE to (input.cacheWriteShort.toDouble() / PER_MILLION * writeShortPrice +
                    input.cacheWriteLong.toDouble() / PER_MILLION * writeLongPrice),
                // completion at the output price + thinking at its own price (equal for text; for image
                // thinking is much cheaper — see LlmPricingProfile.thinkingPricePerMillion).
                CostCategory.OUTPUT to (usage.output.toDouble() / PER_MILLION * outputPrice +
                    usage.thinking.toDouble() / PER_MILLION * thinkingPrice),
                CostCategory.PER_REQUEST to (profile.perRequestUsd ?: 0.0),
            ),
        )
    }

    /** Cost of one cache object (creation at the base input price + storage). ZERO without a profile. */
    fun cacheObject(usage: CacheObjectUsage, profile: LlmPricingProfile?): LlmCostBreakdown {
        if (profile == null) return LlmCostBreakdown.ZERO
        return LlmCostBreakdown.of(
            CostSubject.CACHE_OBJECT,
            mapOf(
                CostCategory.CREATION to usage.createdTokens.toDouble() / PER_MILLION * profile.inputPricePerMillion,
                CostCategory.STORAGE to usage.tokenHours / PER_MILLION * profile.cacheStoragePricePerMillionPerHour,
            ),
        )
    }

    /**
     * Computes cost from pipeline statistics, aggregated per model. Profiles are resolved
     * through [LlmRegistry.findByActualModel] — the provider module is expected to have
     * registered its [Llm] constants on application startup.
     *
     * Map values can be `null`, meaning no pricing profile was registered for that model
     * (callers decide what to do with it: render "no pricing profile", log a warning, etc.).
     */
    fun perModel(stats: PipelineStatistics): Map<String, LlmCostBreakdown?> {
        val byModel = stats.executions
            .filterIsInstance<PromptExecutionEntry>()
            .filter { it.statistics.modelName.isNotEmpty() }
            .groupBy { it.statistics.modelName }

        return byModel.mapValues { (modelName, modelEntries) ->
            val profile = LlmRegistry.findByActualModel(modelName)?.pricingProfile
                ?: return@mapValues null
            // Model cost is a fold of costs: over each ROUND_TRIP and over each cache object.
            // CACHE_CREATE entries are NOT part of RT (their input.total = cache size, not real
            // regular input — it would double-count input). `output.total` is completion, thinking lies separately
            // in `output.thinking` (before 2026-05-08 thinking was not counted — understating reasoning turns).
            val roundTrips = modelEntries.filter { it.nature == ExecutionNature.ROUND_TRIP }
            val rtCost = roundTrips.fold(LlmCostBreakdown.ZERO) { acc, e ->
                val t = e.statistics.tokens
                acc + roundTrip(RoundTripUsage(t.input, t.output.total, t.output.thinking), profile)
            }
            // Cache creation. The size of the created cache lives in the CACHE_CREATE entry (`tokens.input.total`
            // — the same source as the ModelCacheStatistics.createdTokens display). The fallback is for the
            // DB path where CACHE_CREATE rows were abolished (V14) and the size lies in the isCacheCreated
            // round-trip.
            //
            // Storage hours: for PIPELINE-scope caches the declared TTL is not how long the cache really
            // lived. Precision follows the order of `resolveCacheLifetimeMillis`: exact deletion time →
            // elapsed estimate → TTL as the upper bound.
            val creationEntries = modelEntries.filter { it.nature == ExecutionNature.CACHE_CREATE }
            val cacheObjects = if (creationEntries.isNotEmpty()) {
                creationEntries.map { e ->
                    val tokens = e.statistics.tokens.input.total
                    CacheObjectUsage(tokens, tokens.toDouble() * e.statistics.resolveCacheLifetimeMillis(stats) / MILLIS_PER_HOUR)
                }
            } else {
                roundTrips.filter { it.statistics.isCacheCreated == true }.map { e ->
                    val tokens = e.statistics.cachedTokenCount ?: 0L
                    CacheObjectUsage(tokens, tokens.toDouble() * e.statistics.resolveCacheLifetimeMillis(stats) / MILLIS_PER_HOUR)
                }
            }
            cacheObjects.fold(rtCost) { acc, o -> acc + cacheObject(o, profile) }
        }
    }

    fun total(stats: PipelineStatistics): LlmCostBreakdown =
        perModel(stats).values.filterNotNull().fold(LlmCostBreakdown.ZERO) { acc, b -> acc + b }

    /**
     * Cache lifetime used for storage cost, in the precedence exact → elapsed-estimate → TTL:
     *  - exact deletion time ([PipelineStatistics.cacheLifetimeMillis]) — best, rarely available in time;
     *  - elapsed estimate ([PipelineStatistics.cacheEstimatedLifetimeMillis]) — main path for
     *    PIPELINE-scope caches (see its KDoc);
     *  - declared TTL — upper-bound fallback when neither is known.
     */
    private fun PromptExecutionStatistics.resolveCacheLifetimeMillis(stats: PipelineStatistics): Long {
        val id = cacheId
        val exact = id?.let { stats.cacheLifetimeMillis[it] }
        val estimate = id?.let { stats.cacheEstimatedLifetimeMillis[it] }
        return exact ?: estimate ?: (cacheTtlMillis ?: 0L)
    }
}
