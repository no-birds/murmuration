package org.nobirds.murmuration.tooling

/**
 * Marks an instrument whose repeated invocation must NEVER be hard-skipped by the tool-calling
 * agent's repeated-call de-duplication (see `RepeatedCallTracker`) — it always executes on
 * repeat.
 *
 * The de-duplication rests on one assumption: for a pure domain mutator, the same
 * `(name, args)` returning the same result twice proves its effect is applied and
 * idempotent, so a third identical issue is a safe no-op. That assumption is FALSE for two
 * distinct use cases (both opt out of hard-skipping via this annotation, for different reasons):
 * - a *control-flow gate* (e.g. [Stateful.transitionTo]): its result is a conditional check
 *   against mutable external state that OTHER calls change between round-trips (e.g. tasks
 *   being closed) — "same refusal twice" means "still blocked", not "effect applied", so a
 *   stale refusal must never be frozen and replayed forever. This is the exact bug that burned
 *   a planner's whole ACTIONS budget — `transitionTo(WORKING)` refused while a checklist task
 *   was open, then hard-skipped (replaying the stale refusal) on every subsequent round-trip
 *   even after the task had been closed.
 * - a *positional/index-addressed mutator* on a shifting collection (e.g. `removePart(index)`):
 *   the SAME `(index)` legitimately targets a DIFFERENT element on each call (the collection
 *   shifted), yet the returned value can be constant (e.g. `Unit`/"OK") — an identical repeat is
 *   neither a no-op ("changes nothing" would be false — a different element was just mutated)
 *   nor a rejection ("still blocked" would be false — it succeeded).
 *
 * Once a gate's refusal is classified honestly — [Stateful.transitionTo] returns a
 * [TransitionResult] and a [TransitionResult.Failure] is an error via [TransitionResult.success],
 * not via a text-prefix guess — both cases are handled uniformly by the tool-calling agent (see
 * `ToolCallingAgent.executeCall`): an error repeat gets the honest rejection advisory, and a
 * SUCCESSFUL identical repeat of either case is a legitimate re-application that carries NO note.
 *
 * Convention for @NeverSkip tools that return a BARE refusal String (not a [TransitionResult] —
 * e.g. a positional/index gate or a target-id gate that rejects with plain text): the refusal MUST
 * match the `ErrorMatcher` prefix convention (start with `ERROR`, see
 * `ErrorMatcher.resolveDefault`). A bare string is classified as error/ok solely by that prefix, so a
 * refusal that omits it is read as a SUCCESS — and a repeated refusal then gets NO honest-rejection
 * advisory, silently masking that the call is still blocked.
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
annotation class NeverSkip
