package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.prompt.*
import org.nobirds.murmuration.llm.*

/**
 * Who reads the record a tool call leaves behind.
 *
 * This is a property of the tool's CONTRACT, not something observation can measure: only the
 * instrument's author knows who the record it writes is addressed to. Each value answers
 * exactly one class (three kinds of work, two forms of bookkeeping):
 *
 * - [WORLD] — state that outlives the turn: the game world and the builder's own output.
 *   Read by the game and by the next turn. Mutating setters (`updateToken`, `addPart`, `damage`).
 * - [PLAYER_ARTIFACT] — a thing the player will see or hear. `designMap`, `setMusic`.
 * - [CALLING_MODEL] — only this turn's model, data it did not already have. `findPrototypes`.
 * - [RUN_BOOKKEEPING] — the turn itself: its task list and checklist. `addTask`, `completeTask`.
 * - [OWN_PHASE] — the turn's own schedule (not its content) — the ONE class that does NOT pull a
 *   round-trip out of "strictly idle". [Stateful.transitionTo] and nothing else: it says
 *   "this record is read by the schedule", not "this is transitionTo by name".
 */
enum class RecordReader {
    WORLD,
    PLAYER_ARTIFACT,
    CALLING_MODEL,
    RUN_BOOKKEEPING,
    OWN_PHASE,
}

/**
 * Declares who reads the record left by calling this instrument (see [RecordReader]). Absence
 * means "not declared" — a named gap in the analysis, never a fourth silent class.
 *
 * Placed as its OWN annotation, next to `@`[Phases]/`@`[AllPhases]/`@`[NeverSkip]/`@`[CantCloseTask]
 * — not a field on `@`[Instrument]: that annotation's own KDoc puts phase availability,
 * restrictions and validation in separate annotations, and the same holds for this aspect.
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
annotation class LeavesRecord(val reader: RecordReader)

/**
 * Who reads the record left by this instrument (read from `@`[LeavesRecord] on the source
 * function, merged across inheritance the same way `@`[Phases] is — see
 * [InstrumentDeclaration.annotations]). `null` when the instrument's author left it undeclared.
 *
 * When more than one declaration reaches the same instrument (own annotation plus one or more
 * inherited from supertypes — [collectAllAnnotations] appends the function's own annotations
 * first, then each supertype's in `allSuperclasses` order), the FIRST one wins: the closest
 * declaration (the function's own) overrides an inherited one, the same rule by which an
 * implementation overrides an interface in the language itself. Two DIFFERENT inherited
 * declarations (two supertypes, different values) are ordered by `allSuperclasses` and not
 * otherwise resolved by this contract — a named gap.
 */
val InstrumentDeclaration<*>.recordReader: RecordReader?
    get() = annotations.filterIsInstance<LeavesRecord>().firstOrNull()?.reader
