package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.prompt.*
import org.nobirds.murmuration.llm.*

/**
 * Marker interface for agent states.
 */
interface AgentState

/**
 * Interface for tools/builders that maintain an [AgentState].
 */
interface Stateful<S : AgentState, B : Any> {
    val currentState: S
    val stateMachine: StateMachineDefinition<S, B>

    /**
     * Phases applied to user-declared instruments without their own [Phases]/[AllPhases]
     * at agent setup time. Empty (default) keeps the legacy "available in every phase"
     * semantics. Implementor builders typically return `listOf(ImplementorAgentState.WORKING)`
     * so that mutating setters are confined to WORKING and the SELF_REVIEW phase becomes
     * read-only (only `addTask`/`transitionTo` are reachable there).
     */
    val defaultUserToolPhases: List<S> get() = emptyList()

    /**
     * Dynamic (round-trip-time) narrowing of the tool allowlist, applied ON TOP of the static
     * [Phases] / [defaultUserToolPhases] gate. [staticallyAllowed] is the set of instrument
     * names the static gate already permits in [state]; this returns the subset actually
     * callable on the CURRENT round-trip given the builder's live substate. Default — identity
     * (no narrowing).
     *
     * Static [Phases] answers "is this tool reachable in this phase at all" (compile-time, per
     * state); this hook answers "given the builder's mutable substate, is it reachable RIGHT
     * NOW". It is the only DYNAMIC tool gate in the framework, consulted in BOTH the rendered
     * "Tools available" block and the runtime call gate ([ToolsExecutor.executeCall]), so a
     * narrowed-out tool is enforced (returns the same "not available" error as a wrong-phase
     * call), not merely un-advertised.
     *
     * Because the narrowing is recomputed each round-trip (at its start for the block, and at
     * call time for the gate), a builder can safely widen it back as its substate changes — a
     * lock imposed for one round-trip lifts on the next once the condition clears.
     *
     * Example ([org.nobirds...TaskAwareBuilder]): once every task is closed and only a valid
     * `selfCheck` is owed (ON_FINDINGS WORKING substate), the working mutators are narrowed out —
     * re-issuing one there only spawns a duplicate; the allowlist collapses to the finishing
     * tools until `selfCheck` reopens work via fix-tasks.
     */
    @NotAnInstrument
    fun availableTools(state: S, staticallyAllowed: Set<String>): Set<String> = staticallyAllowed

    /**
     * Requests a transition to the next state. Returns a [TransitionResult] — the framework
     * renders its [TransitionResult.message] as the tool result the LLM sees, and classifies a
     * [TransitionResult.Failure] as an error/refusal HONESTLY (via [TransitionResult.success]),
     * without depending on the message text matching any [ErrorMatcher] prefix convention. This
     * is what lets the repeated-call tracker tell a real repeated refusal (→ refusal advisory)
     * apart from a legitimate repeated successful transition (→ no advisory): a refusal whose
     * wording carries no error prefix (e.g. "Close the task, then retry") is still recognised as
     * a refusal because it is a `Failure`, and a byte-identical `Success` repeated twice (e.g.
     * WORKING→REVIEW happening twice in one run) is no longer mislabelled "REJECTED".
     *
     * Contract (framework-wide, «transitions apply between round-trips»): a call made during
     * a round-trip only REGISTERS the intent — the gate is evaluated immediately (the model
     * gets an honest Success/Failure), but the state change and the transition's
     * [Transition.effect] are applied ONCE, at the end of the round-trip (in
     * [onAfterToolCalls]), AFTER every other tool call of the same response has run in the
     * current phase. This removes the class of "tool not available in phase" rejects that
     * used to hit the tail of a batch when a mid-batch transition flipped the phase. See
     * [TaskAwareBuilder]-style implementations for the canonical pending/apply machinery; the
     * trivial [DefaultStatefulImpl] defers the same way (the call registers a pending target,
     * [onAfterToolCalls] applies it) — an immediate inline apply would also defeat the
     * framework's forced error-visibility hold (see [ToolCallingAgent.errorMatcher]): an
     * errored batch carrying a terminal `transitionTo` would end the invocation before the
     * forced round-trip could run.
     *
     * Annotated with [AllPhases] so it stays reachable from every state — without this
     * marker the Stateful default (e.g. WORKING-only for implementors) would lock the
     * agent inside SELF_REVIEW with no way to advance.
     */
    @AllPhases
    @Instrument("Transition to another state (phase) of work. Call it once the current phase's work is complete — as the LAST call of the SAME response that finishes that work (when the phase rules allow the transition). The switch is applied at the END of this response: every other call in the same response still runs in the CURRENT phase, so it is safe to finish the phase's work and request the transition together. Do not spend a separate response on a lone transition call, and do not place new work AFTER the transition expecting the new phase.")
    @CantCloseTask
    @NeverSkip
    @LeavesRecord(RecordReader.OWN_PHASE)
    suspend fun transitionTo(@InstrumentParam("Target state") nextState: S): TransitionResult

    /**
     * Hook the framework calls AFTER all tool calls of the current round-trip have run.
     * This is the SINGLE point where state transitions are applied (see [transitionTo]):
     * an explicitly requested transition wins over any auto-transition trigger; otherwise
     * the builder may transition itself based on its own state (for example, the absence of
     * unfinished tasks). Applying here — between round-trips — guarantees that every tool
     * call of the batch already ran in the phase the round-trip started in.
     *
     * A request to transition into the CURRENT phase is not "an explicitly requested transition"
     * here: it changes no phase, so it must neither spend the response's single transition nor
     * cancel an armed auto-trigger (an undeclared one is a no-op — see [checkTransition]; a
     * declared one only runs its effect).
     *
     * [stateAtStart] is the state in which the round-trip started (before tool calls). The
     * builder can use it for safety-net auto-transitions that must only fire when the model
     * has already seen the current phase's prompt (e.g. the SELF_REVIEW → next auto-advance
     * fires only when the round-trip started in SELF_REVIEW).
     *
     * Default — no-op.
     */
    @NotAnInstrument
    suspend fun onAfterToolCalls(stateAtStart: S) {}

    /**
     * Hook the framework calls at the START of every round-trip, before the phase
     * description is rendered. Lets the builder snapshot per-round-trip context (e.g. the
     * state the round-trip started in) without relying on call-order conventions inside
     * tool handlers. Default — no-op.
     */
    @NotAnInstrument
    suspend fun onRoundTripStart(state: S) {}

    /**
     * Hook the framework calls when a round-trip is ABORTED and will be re-attempted: the model
     * call failed and the loop's round-trip retry ([org.nobirds.murmuration.retry.RetryPolicy])
     * decided to replay the SAME productive turn on the preserved history. Nothing of the failed
     * attempt was appended to the conversation, so from the model's point of view the round-trip
     * never happened — but the builder may already hold per-round-trip work the aborted batch
     * produced: tool jobs dispatched before the stream broke run to completion (see
     * `ToolsExecutor.executeTurn`'s `finally`), and the retry path skips [onAfterToolCalls]
     * entirely, so anything that batch buffered for the END of the round-trip is never consumed.
     *
     * This is the builder's single chance to drop such a buffer, so the replayed round-trip does
     * not execute it twice (or execute it for a batch the model will now re-issue). Symmetric to
     * [onForcedTermination] — a salvage hook, not a state transition point: the phase is untouched
     * and the loop continues. Whatever is dropped must leave a trace the model can see, exactly as
     * on a forced termination. [cause] is the failure that triggered the retry. Default — no-op.
     */
    @NotAnInstrument
    suspend fun onRoundTripAborted(state: S, cause: Throwable) {}

    /**
     * Hook the framework calls when the agent loop exits WITHOUT reaching a terminal state:
     * the call/round-trip budget was exhausted mid-work (a [CallLimitStrategy] GracefulStop, or
     * `maxTurns` ran out). Unlike a clean terminal finish — where the state machine gates the
     * terminal transition on all tasks being closed — this exit ABANDONS whatever work was still
     * open. The builder gets its single chance to salvage it (e.g. escalate the still-open tasks
     * so they are not silently lost when the loop just `return`s). [reason] is a short
     * human-readable cause. Default — no-op.
     */
    @NotAnInstrument
    suspend fun onForcedTermination(state: S, reason: String) {}

    /**
     * Optional hook returning a snapshot block the framework should re-emit on every
     * round-trip's phase description, marked so that older copies are pruned from the
     * chat history (only the latest occurrence ships to the model).
     *
     * Returning `null` (default) disables the feature — the agent's chat history stays
     * append-only as before. Override when the builder maintains a mutating output state
     * that the model needs to see in full *each* round-trip but where stale copies are
     * pure token waste (e.g. accumulated entities under construction).
     *
     * The pruning itself is applied by a `PromptExecutor` decorator just before the prompt
     * leaves for the provider, so this hook is safe to enable independently of any cache
     * configuration — appended history remains intact for debug / trace logging.
     */
    @NotAnInstrument
    fun currentReplaceableSnapshot(): SnapshotBlock? = null

    /**
     * Snapshot returned by [currentReplaceableSnapshot] — the [sectionId] groups
     * occurrences so the executor knows which earlier copies to drop, and [text] is the
     * fresh body to embed as a single `TextPromptPart` in the phase description.
     */
    data class SnapshotBlock(val sectionId: String, val text: String)
}

/**
 * Result of a state transition attempt.
 */
sealed interface TransitionResult {
    val success: Boolean
    val message: String

    data class Success(override val message: String) : TransitionResult {
        override val success: Boolean = true
    }

    data class Failure(override val message: String) : TransitionResult {
        override val success: Boolean = false
    }
}

/**
 * Definition of a state machine for an agent.
 */
interface StateMachineDefinition<S : AgentState, B : Any> {
    val initialState: S
    val allTransitions: List<Transition<S, B>>
    val allNodes: Map<S, StateMachineNode<S, B>>
    fun getNode(state: S): StateMachineNode<S, B>

    /**
     * Validates the transition via its check and, on success, runs the transition's
     * [Transition.effect] (if any) BEFORE returning. The effect is part of the transition:
     * a throwing effect aborts the transition and propagates to the agent loop.
     */
    suspend fun transitionTo(currentState: S, nextState: S, context: B): TransitionResult

    /**
     * Runs ONLY the transition's [Transition.check] against [currentState]/[nextState],
     * WITHOUT running any [Transition.effect] and without changing state. Used to gate an
     * explicit `transitionTo` at CALL time (mid round-trip) so the model gets an honest,
     * immediate Success/Failure, while the actual apply (check + effect + state change via
     * [transitionTo]) is deferred to the end of the round-trip. Same no-op / not-defined /
     * not-allowed semantics as [transitionTo], minus the effect.
     *
     * Default implementation covers the standard [StateMachineNode] topology; provided as a
     * default so existing [StateMachineDefinition] implementations keep compiling.
     */
    fun checkTransition(currentState: S, nextState: S, context: B): TransitionResult {
        val node = getNode(currentState)
        if (currentState == nextState) {
            val declared = (node as? StateMachineNode.State)?.transitions?.any { it.targetState == nextState } == true
            if (!declared) {
                // Friendly no-op, NOT an error: the phase is already the requested one, so nothing
                // is applied and NOTHING is spent — the call does not consume this response's single
                // transition (see the deferred-apply builders). The way forward is spelled out from
                // the MACHINE itself (declared targets of this node), so the model does not have to
                // spend a whole round-trip discovering it.
                val targets = (node as? StateMachineNode.State)?.transitions
                    ?.map { it.targetState.toString() }
                    ?.distinct()
                    .orEmpty()
                val hint = if (targets.isEmpty()) {
                    "There is nothing to transition to from here — just finish this phase's work."
                } else {
                    "Once this phase's work is done, call transitionTo(${targets.joinToString(" | ")}) — " +
                        "you may do it in THIS same response."
                }
                return TransitionResult.Success(
                    "Already in $nextState — no-op, nothing was applied and this response's transition is NOT spent. $hint"
                )
            }
        }
        if (node is StateMachineNode.State) {
            val transition = node.transitions.find { it.targetState == nextState }
                ?: return TransitionResult.Failure("Transition from $currentState to $nextState is not defined.")
            return transition.check(context)
        }
        return TransitionResult.Failure("Current state $currentState does not allow transitions.")
    }
}

/**
 * Configuration for a specific state.
 */
sealed class StateMachineNode<S : AgentState, B : Any> {
    abstract val state: S
    abstract val summary: String
    abstract val description: (B) -> String

    data class State<S : AgentState, B : Any>(
        override val state: S,
        override val summary: String,
        override val description: (B) -> String,
        val transitions: List<Transition<S, B>>,
    ) : StateMachineNode<S, B>()

    data class Terminal<S : AgentState, B : Any>(
        override val state: S,
        override val summary: String,
        override val description: (B) -> String
    ) : StateMachineNode<S, B>()
}

/**
 * Describes a transition from one state to another.
 *
 * A transition is validation ([check]) + optional work ([effect]) + the state change.
 * The effect runs only after a successful check and is suspend-capable: a delegating
 * transition may run real work (e.g. execute a sub-agent over a staged batch) as PART of
 * the transition — the model calls `transitionTo`, the work happens inside the tool call,
 * and the next round-trip starts in the target state with the results visible in the
 * snapshot. A throwing effect aborts the agent invocation (same as a failing tool).
 */
data class Transition<S : AgentState, B : Any>(
    val targetState: S,
    val toolName: String,
    val description: String,
    val check: (B) -> TransitionResult = { TransitionResult.Success("Successful transition to $targetState") },
    val effect: (suspend (B) -> Unit)? = null,
)

enum class DefaultAgentState : AgentState { WORKING, FINISHED }

/**
 * Role mapping of the task-loop pattern onto a concrete agent state type. The pattern
 * (tasks → batches → auto self-review → rework budget) needs to know which state plays
 * which role, while every agent kind may define its own state enum. Implemented by the
 * enum's companion object, so the mapping is declared once next to the states and call
 * sites read naturally: `roles = TurnPlannerState`.
 *
 * Naming convention: role states are canonically named `WORKING` / `SELF_REVIEW` /
 * `FINISHED` in every enum (plus any custom states) — string-based `@Phases` gates
 * rely on these names.
 */
interface TaskStateRoles<S : AgentState> {
    val working: S
    val selfReview: S
    val finished: S
}

/**
 * Implementor states that support an optional self-review phase.
 *
 * - [WORKING]: the main phase; the agent closes tasks set by the planner.
 * - [SELF_REVIEW]: the agent re-reads its own output against the domain checklist and
 *   fixes anything it noticed BEFORE moving to FINISHED. Optional; when disabled the
 *   state machine reduces to `WORKING → FINISHED`.
 * - [FINISHED]: terminal state.
 */
enum class ImplementorAgentState : AgentState {
    WORKING, SELF_REVIEW, FINISHED;

    companion object : TaskStateRoles<ImplementorAgentState> {
        override val working: ImplementorAgentState = WORKING
        override val selfReview: ImplementorAgentState = SELF_REVIEW
        override val finished: ImplementorAgentState = FINISHED

        // Compile-time string constants to be used inside @Phases annotations
        // (annotation parameters must be compile-time, .name is not).
        const val WORKING_NAME = "WORKING"
        const val SELF_REVIEW_NAME = "SELF_REVIEW"
        const val FINISHED_NAME = "FINISHED"
    }
}

/**
 * Creates a default state machine (Working -> Finished).
 */
fun <B : Any> defaultStateMachine(): StateMachineDefinition<DefaultAgentState, B> = stateMachine(initial = DefaultAgentState.WORKING) {
    state(DefaultAgentState.WORKING, "Main work", "Performing the agent's primary task.") {
        transition(
            to = DefaultAgentState.FINISHED,
            name = "finish",
            description = "Finish the work and return the final result."
        )
    }
    terminal(DefaultAgentState.FINISHED, "Finished", "Work is complete; the result has been produced.")
}

/**
 * Creates a default [Stateful] implementation for a tool.
 */
fun <B : Any> defaultStateful(): Stateful<DefaultAgentState, B> =
    DefaultAgentStatefulImpl(defaultStateMachine())

private class DefaultAgentStatefulImpl<B : Any>(
    stateMachine: StateMachineDefinition<DefaultAgentState, B>
) : DefaultStatefulImpl<DefaultAgentState, B>(stateMachine)

private open class DefaultStatefulImpl<S : AgentState, B : Any>(
    override val stateMachine: StateMachineDefinition<S, B>
) : Stateful<S, B> {
    private var state = stateMachine.initialState
    override val currentState: S get() = state

    /**
     * Deferred applier (framework contract, see [Stateful.transitionTo]): the call only registers
     * the pending target; [onAfterToolCalls] applies it at the end of the round-trip. The pending
     * target deliberately SURVIVES a forced error-visibility hold (which skips [onAfterToolCalls]):
     * it then applies at the end of the next non-held round-trip, so «the transition lands only
     * after a clean round» without requiring the model to re-issue the call.
     */
    private var pending: S? = null

    override suspend fun transitionTo(nextState: S): TransitionResult =
        TransitionResult.Success("Successful transition to $nextState").also { pending = nextState }

    override suspend fun onAfterToolCalls(stateAtStart: S) {
        pending?.let { state = it }
        pending = null
    }

    override suspend fun onRoundTripAborted(state: S, cause: Throwable) {
        // The failed attempt never reached the conversation — drop whatever it armed, the model
        // will re-issue the batch on the retry.
        pending = null
    }
}

/**
 * DSL for creating a state machine definition.
 */
fun <S : AgentState, B : Any> stateMachine(
    initial: S,
    block: StateMachineBuilder<S, B>.() -> Unit
): StateMachineDefinition<S, B> {
    val builder = StateMachineBuilder<S, B>(initial)
    builder.block()
    return builder.build()
}

class StateMachineBuilder<S : AgentState, B : Any>(private val initial: S) {
    private val nodes = mutableMapOf<S, StateMachineNode<S, B>>()

    fun state(state: S, summary: String, description: String, block: StateNodeBuilder<S, B>.() -> Unit) =
        state(state, summary, { description }, block)

    fun state(state: S, summary: String, description: (B) -> String, block: StateNodeBuilder<S, B>.() -> Unit) {
        val builder = StateNodeBuilder<S, B>(state, summary, description)
        builder.block()
        nodes[state] = builder.build()
    }

    fun terminal(state: S, summary: String, description: String) =
        terminal(state, summary) { description }

    fun terminal(state: S, summary: String, description: (B) -> String) {
        nodes[state] = StateMachineNode.Terminal(state, summary, description)
    }

    fun build(): StateMachineDefinition<S, B> = StateMachineDefinitionImpl(initial, nodes)
}

class StateNodeBuilder<S : AgentState, B : Any>(private val state: S, private val summary: String, private val description: (B) -> String) {
    private val transitions = mutableListOf<Transition<S, B>>()

    fun transition(
        to: S,
        name: String,
        description: String,
        effect: (suspend (B) -> Unit)? = null,
        check: (B) -> TransitionResult = { TransitionResult.Success("Successful transition to $to") },
    ) {
        transitions.add(Transition(to, name, description, check, effect))
    }

    fun transitionIf(to: S, name: String, description: String, errorMessage: String, condition: (B) -> Boolean) {
        transition(to, name, description) {
            if (condition(it)) TransitionResult.Success("Successful transition to $to")
            else TransitionResult.Failure(errorMessage)
        }
    }

    fun transitionIf(to: S, name: String, description: String, condition: (B) -> Boolean, errorMessage: (B) -> String) {
        transition(to, name, description) {
            if (condition(it)) TransitionResult.Success("Successful transition to $to")
            else TransitionResult.Failure(errorMessage(it))
        }
    }

    fun build(): StateMachineNode<S, B> = StateMachineNode.State(state, summary, description, transitions)
}

private class StateMachineDefinitionImpl<S : AgentState, B : Any>(
    override val initialState: S,
    override val allNodes: Map<S, StateMachineNode<S, B>>
) : StateMachineDefinition<S, B> {
    override val allTransitions: List<Transition<S, B>> = allNodes.values
        .filterIsInstance<StateMachineNode.State<S, B>>()
        .flatMap { it.transitions }
        .distinctBy { it.toolName }

    override fun getNode(state: S): StateMachineNode<S, B> = allNodes[state] ?: throw IllegalArgumentException("Unknown state: $state")

    override suspend fun transitionTo(currentState: S, nextState: S, context: B): TransitionResult {
        // Gate first (shared with [checkTransition]) — no effect on failure or on an
        // undeclared self-transition no-op.
        val result = checkTransition(currentState, nextState, context)
        if (!result.success) return result
        // Declared transition (including a DECLARED self-transition, e.g. "execute the staged
        // batch and keep planning in WORKING") — run its effect. An undeclared self-transition
        // returned Success above but has no Transition object, so no effect runs.
        val node = getNode(currentState)
        (node as? StateMachineNode.State)?.transitions?.find { it.targetState == nextState }
            ?.effect?.invoke(context)
        return result
    }
}
