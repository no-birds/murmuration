package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.agent.AgentEventListener
import org.nobirds.murmuration.agent.CallLimitDecision
import org.nobirds.murmuration.agent.CallLimitStrategy
import org.nobirds.murmuration.agent.ExecutionContext

/**
 * Observability capability for the tool-calling agent: round-trips, individual tool calls,
 * state transitions, budget exhaustion. All methods are `suspend` (listeners may write to
 * async sinks) and all default to no-op (implementers override only what they care about).
 *
 * NOT intended for mutating builder state — use [Stateful.onAfterToolCalls] for that.
 * This is observation only.
 *
 * States are delivered as untyped [AgentState]: the framework dispatches to all
 * subscribers regardless of the agent's concrete state type. A listener that needs the
 * concrete type does `state as MyAgentState` itself, knowing which agents it is
 * registered with.
 */
interface ToolCallingListener : AgentEventListener {
    /** Called before each LLM round-trip (one iteration of `executeWithTools`). */
    suspend fun onBeforeTurn(state: AgentState) {}

    /** Called after a round-trip (after tool calls have been applied and the builder safety-net has run). */
    suspend fun onTurnFinished(state: AgentState) {}

    /** Called before executing a single tool call within a round-trip. */
    suspend fun onBeforeToolCall(state: AgentState, name: String, args: Map<String, Any?>) {}

    /**
     * Called after a successful tool call. [stateBefore] is the state the tool call ran in.
     * With deferred transitions ([Stateful.transitionTo] only registers the intent mid-batch;
     * it is applied at the end of the round-trip in [Stateful.onAfterToolCalls]) this equals
     * [state] for the standard builders — the phase does not flip inside a call. It is kept as
     * a distinct parameter for Stateful implementations that apply transitions immediately.
     */
    suspend fun onAfterToolCall(state: AgentState, stateBefore: AgentState, name: String, result: Any?) {}

    /** Called when a tool call throws. */
    suspend fun onToolCallError(state: AgentState, name: String, error: Throwable) {}

    /**
     * Called once per round-trip when the state changed across it — the framework diffs the
     * state the round-trip started in against the state after the between-round-trips apply
     * point ([Stateful.onAfterToolCalls]). Fires for both deferred appliers (state flips at the
     * apply point) and immediate appliers (state flipped inside a tool call). Not fired when a
     * transition was requested but cancelled/not applied (state unchanged across the round-trip).
     */
    suspend fun onStateTransition(from: AgentState, to: AgentState) {}

    /** Called when the round-trip counter is exhausted (`maxTurns`) and the agent returns a partial result. */
    suspend fun onMaxTurnsExhausted(state: AgentState) {}

    /**
     * Called when [CallLimitStrategy] returned [CallLimitDecision.GracefulStop] and the
     * agent exits the loop with a partial result. [reason] is the message produced by
     * the strategy.
     */
    suspend fun onCallLimitExhausted(state: AgentState, reason: String) {}
}

/**
 * Invokes [block] on every listener registered in [ExecutionContext] that implements
 * [ToolCallingListener]. The framework iterates the registry through
 * [ExecutionContext.forEachAgentListener]; this helper just adds the type check.
 */
suspend fun ExecutionContext.dispatchToolCalling(
    block: suspend ToolCallingListener.() -> Unit,
) {
    forEachAgentListener { listener ->
        if (listener is ToolCallingListener) listener.block()
    }
}
