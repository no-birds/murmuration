package org.nobirds.murmuration.llm

import org.nobirds.murmuration.prompt.ThinkingLevel

/**
 * Model FEATURES and provider CONSTRAINTS, carried as a set of sealed descriptors on
 * [Llm.capabilities] — orthogonal to content modalities ([Llm.inputModalities] /
 * [Llm.outputModalities], also sealed sets).
 *
 * A set of sealed descriptors, not capability interfaces: models are config-resolved (the static
 * type is always [Llm]), so a `fun foo(llm: Caching)`-style static expectation never applies — every
 * query is dynamic regardless. A set keeps that dynamic lookup, carries per-capability data, and lets
 * a new capability be a new subtype — no [Llm]-interface bloat, no `object : Llm by …` delegation on
 * every constant. Query via [capability]:
 *
 * ```
 * prompt.llm.capability<Caching>()?.minTokens
 * ```
 *
 * Facet-rich capabilities hold their own sealed set (see [Thinking.modes]); a flagless one is a
 * `data object` ([ToolCalling]).
 */
sealed interface Capability

/**
 * Provider-side prompt caching: `cache.create` is rejected for content below [minTokens] tokens
 * (e.g. Vertex Gemini 3.x: 4096; 2.5: 1024).
 */
data class Caching(val minTokens: Int) : Capability

/**
 * Reasoning/thinking. [modes] (a sealed set) describes how it's controlled — extend by adding a
 * [ThinkingMode], each carrying its own data. Whether the trace is emitted is a separate fact:
 * `Modality.Thoughts` on [Llm.outputModalities].
 */
data class Thinking(val modes: Set<ThinkingMode>) : Capability

/**
 * Function/tool calling (`ToolsPromptSettings` / `ToolCallingAgent`). Implies `Modality.Text` on the
 * output axis (a precondition carried by [Llm.outputModalities], not re-encoded here).
 */
data object ToolCalling : Capability

sealed interface ThinkingMode {
    /**
     * Controlled by a tier (`thinkingLevel`: MINIMAL/LOW/MEDIUM/HIGH) — Gemini 3.x / Pro-class.
     * [supported] — the tiers the provider actually accepts for this model; `null` = all four.
     * Some models reject specific tiers with a 400 (e.g. `gemini-3.1-flash-lite-image`: LOW is
     * rejected — 400 — while MINIMAL and HIGH are both supported, per live probes 2026-07-15)
     * — declare the fact here so callers can clamp instead of crashing.
     */
    data class Level(val supported: Set<ThinkingLevel>? = null) : ThinkingMode

    /** Controlled by a numeric token budget (`thinkingBudget`) — e.g. Gemini 2.5 Flash. */
    data class Budget(val maxBudget: Int? = null) : ThinkingMode
}

/** The model's capability of type [T], or `null` if it has none. See [Capability]. */
inline fun <reified T : Capability> Llm.capability(): T? = capabilities.filterIsInstance<T>().firstOrNull()

/**
 * [level] if this model accepts it, otherwise the closest supported tier below it (or the
 * lowest supported one). Models without a declared [ThinkingMode.Level] constraint — including
 * models with no [Thinking] capability at all — pass [level] through unchanged: absence of a
 * declaration means "no known constraint", not "no thinking".
 */
fun Llm.clampThinkingLevel(level: ThinkingLevel): ThinkingLevel {
    val supported = capability<Thinking>()?.modes
        ?.filterIsInstance<ThinkingMode.Level>()?.firstOrNull()
        ?.supported ?: return level
    if (level in supported) return level
    return supported.filter { it < level }.maxOrNull() ?: supported.min()
}
