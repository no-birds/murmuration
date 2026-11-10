package org.nobirds.murmuration.llm

/** What a cost category is billed for: one round-trip, or one Gemini cache object. */
enum class CostSubject { ROUND_TRIP, CACHE_OBJECT }

/**
 * The closed set of cost categories (USD). A stored cost is always a total function "category -> USD"
 * ([LlmCostBreakdown.amount] / [LlmCostBreakdown.of]), so a writer that walks [entries] cannot silently
 * lose a category: adding a value here breaks every exhaustive `when` over it and the schema test
 * that requires a column per category of its [subject].
 *
 * The enum names are the column-name stems: `cost_<name in lower case>_usd`.
 */
enum class CostCategory(val subject: CostSubject) {
    REGULAR_INPUT(CostSubject.ROUND_TRIP),
    CACHED_INPUT(CostSubject.ROUND_TRIP),
    CACHE_WRITE(CostSubject.ROUND_TRIP),
    OUTPUT(CostSubject.ROUND_TRIP),
    PER_REQUEST(CostSubject.ROUND_TRIP),

    /** Gemini cache object creation. */
    CREATION(CostSubject.CACHE_OBJECT),

    /** Gemini cache object storage. */
    STORAGE(CostSubject.CACHE_OBJECT),
    ;

    /** Name of the column holding this category, in the table of its [subject]. */
    val columnName: String get() = "cost_${name.lowercase()}_usd"
}
