package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.prompt.*
import org.nobirds.murmuration.llm.*

/**
 * Generates a global overview of the state machine structure for the system prompt.
 */
fun StateMachineDefinition<*, *>.generateGlobalInstruction(context: Any): String {
    @Suppress("UNCHECKED_CAST")
    val definition = this as StateMachineDefinition<AgentState, Any>
    val sb = StringBuilder()
    sb.append("OVERALL PROCESS STRUCTURE (STATE MACHINE):\n")
    definition.allNodes.forEach { (state, node) ->
        sb.append("- State: $state\n")
        sb.append("  Description: ${node.summary}\n")
        if (node is StateMachineNode.State) {
            node.transitions.forEach { transition ->
                sb.append("  -> Transition to ${transition.targetState}: ${transition.description}\n")
            }
        } else {
            sb.append("  (Terminal — work complete)\n")
        }
    }
    return sb.toString()
}

/**
 * Generates the per-turn instruction describing the current state. The list of tools
 * available in this phase is emitted separately by `ToolCallingAgent.appendPhaseDescription`
 * (as a text block in the same user-message), so this helper stays focused on the phase
 * narrative itself.
 */
fun StateMachineDefinition<*, *>.generateCurrentStateInstruction(
    currentState: AgentState,
    context: Any,
): String {
    @Suppress("UNCHECKED_CAST")
    val definition = this as StateMachineDefinition<AgentState, Any>
    val sb = StringBuilder()
    val node = definition.getNode(currentState)

    sb.append("CURRENT STATE (PHASE): $currentState\n")
    sb.append("CURRENT PHASE DESCRIPTION: ${node.description(context)}\n")

    if (node is StateMachineNode.Terminal) {
        sb.append("\nThis is a terminal state. Work is complete.\n")
    }
    // Intentionally no AVAILABLE TRANSITIONS / "MUST call transitionTo" footer: the
    // global state-machine instruction (rendered once per session) already lists all
    // transitions, and the per-phase description carries the authoritative rule for
    // *when* and *whether* to call transitionTo. A blanket framework footer here
    // tended to override per-phase guidance like SELF_REVIEW's "do not call
    // transitionTo(SELF_REVIEW)".

    return sb.toString()
}

/**
 * Convenience helper that concatenates the global structure overview with the per-state
 * instruction. Prefer the split methods directly.
 */
fun StateMachineDefinition<*, *>.generateInstruction(currentState: AgentState, context: Any): String {
    return generateGlobalInstruction(context) + "\n" + generateCurrentStateInstruction(currentState, context)
}
