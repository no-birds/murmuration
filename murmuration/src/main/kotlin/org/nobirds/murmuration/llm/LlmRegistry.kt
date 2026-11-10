package org.nobirds.murmuration.llm

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.prompt.*

import java.util.concurrent.CopyOnWriteArrayList

/**
 * Global registry of known [Llm] constants. Provider modules (for example
 * `:murmuration-gemini`) register their models on application startup; everything else
 * (chiefly [LlmCostCalculator]) resolves an [Llm] by the actual model name from the API
 * through [findByActualModel].
 *
 * Registration is explicit: do it in `Application.kt` or another entry point before any
 * pipeline starts producing statistics. We do not rely on lazy class-loading of provider
 * objects to populate the registry — an empty registry at the moment of the first lookup
 * silently loses data.
 */
object LlmRegistry {
    private val models = CopyOnWriteArrayList<Llm>()

    fun register(llm: Llm) {
        models += llm
    }

    fun register(llms: Iterable<Llm>) {
        models += llms
    }

    /**
     * Resolves an [Llm] by the actual model name from the API response. On a collision,
     * returns the constant with the longest matching `model` — needed when both
     * `gemini-3-flash` and `gemini-3-flash-preview` are registered: for an actual name of
     * `gemini-3-flash-preview-04-17`, the longer one wins.
     */
    fun findByActualModel(actualModelName: String): Llm? =
        models.filter { it.matches(actualModelName) }
            .maxByOrNull { it.model.length }

    fun all(): List<Llm> = models.toList()

    /** Test-only: drop registered models. */
    internal fun clear() {
        models.clear()
    }
}
