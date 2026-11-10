package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.prompt.*
import org.nobirds.murmuration.llm.*

/**
 * Restricts the state-machine phases in which an instrument is available. If absent or
 * empty, the instrument is available in every phase. This is part of the tool-calling
 * agent contract (state-machine-driven flow), not a generic agent feature.
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
annotation class Phases(vararg val phases: String)

/**
 * Marks an instrument as available in every phase regardless of
 * [Stateful.defaultUserToolPhases]. Without this marker an instrument with no [Phases]
 * annotation gets the Stateful default applied at agent setup time (see
 * [withDefaultPhases]); a few framework instruments (notably [Stateful.transitionTo])
 * need to be reachable from every state, so they opt out via [AllPhases].
 *
 * Choose between annotations like this:
 *  - `@Phases("X", "Y")` — only in listed states.
 *  - `@AllPhases`        — always available, default never applies.
 *  - neither             — Stateful default fills the phases list at setup time. If the
 *    default is also empty, the instrument is available in every phase (back-compat).
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
annotation class AllPhases

/**
 * Phases in which this instrument is available (read from `@`[Phases] on the source
 * function). Empty list means available in every phase.
 */
val InstrumentDeclaration<*>.phases: List<String>
    get() = annotations.filterIsInstance<Phases>().firstOrNull()?.phases?.toList() ?: emptyList()

/**
 * `true` when the instrument explicitly declares its phase set (via [Phases] or [AllPhases]).
 * Used by [withDefaultPhases] to decide whether to apply the Stateful default.
 */
val InstrumentDeclaration<*>.hasExplicitPhases: Boolean
    get() = annotations.any { it is Phases || it is AllPhases }

/**
 * Returns a list where instruments **without** explicit phase declarations get a
 * synthetic `@Phases` annotation built from [phases]. Instruments that already declared
 * their own [Phases] or carry [AllPhases] are returned unchanged. No-op when [phases]
 * is empty.
 *
 * Called by [ToolCallingAgent] at setup time with [Stateful.defaultUserToolPhases] so
 * tool-calling agents can default to e.g. "WORKING-only" semantics for user setters
 * without aspect-side annotations on every method.
 */
fun List<InstrumentDeclaration<*>>.withDefaultPhases(
    phases: List<String>,
): List<InstrumentDeclaration<*>> {
    if (phases.isEmpty()) return this
    val anno = Phases(phases = phases.toTypedArray())
    return map { if (it.hasExplicitPhases) it else it.withAdditionalAnnotations(listOf(anno)) }
}
