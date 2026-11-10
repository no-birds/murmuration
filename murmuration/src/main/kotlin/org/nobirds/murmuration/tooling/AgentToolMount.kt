package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.prompt.InstrumentDeclaration
import org.nobirds.murmuration.prompt.InstrumentsDeclaration
import org.nobirds.murmuration.prompt.PromptSettings
import org.nobirds.murmuration.prompt.ToolsPromptSettings

/**
 * The single writer of "what is mounted on a tool-calling agent".
 *
 * Built once from `(tool, stateful)`. Everything that needs the mounted set reads it from here:
 * the agent (phase description, provider settings), and tests/probes that must measure exactly
 * what reaches the provider.
 *
 * @property declarations exactly what goes to the provider: the tool's declaration followed by
 *   the stateful's declaration (absent when `stateful === tool`).
 */
class AgentToolMount private constructor(
    val declarations: List<InstrumentsDeclaration<*>>,
    private val defaultPhases: List<String>,
) {
    /** The same instruments as [declarations], with [Stateful.defaultUserToolPhases] applied. */
    val instruments: List<InstrumentDeclaration<*>> =
        declarations.flatMap { it.instruments }.withDefaultPhases(defaultPhases)

    /**
     * Prompt settings carrying the mounted tools. **Replaces** `tools` (never appends) and keeps the
     * other fields of [base]. A generator must not declare tools in its prompt: agent tools are
     * declared through the builder only, so non-empty `base.tools` is rejected.
     */
    fun toolsSettings(base: PromptSettings?): ToolsPromptSettings {
        val current = base as? ToolsPromptSettings ?: ToolsPromptSettings(emptyList())
        require(current.tools.isEmpty()) {
            "Prompt settings of a tool-calling agent must not declare tools (found ${current.tools.size}); " +
                "agent tools are mounted only through the builder"
        }
        return current.copy(tools = declarations)
    }

    companion object {
        /**
         * The single rule for "which stateful stands behind [tool]": the tool itself when it is a [Stateful],
         * otherwise [defaultStateful] (every plain tool/classifier). Feed the result to [of].
         */
        fun statefulOf(tool: Any): Stateful<*, *> = tool as? Stateful<*, *> ?: defaultStateful<Any>()

        fun of(tool: Any, stateful: Stateful<*, *>): AgentToolMount {
            val factory = InstrumentFactory()
            fun declare(target: Any): InstrumentsDeclaration<*> =
                target as? InstrumentsDeclaration<*> ?: factory.create(target)
            val declarations = buildList {
                add(declare(tool))
                if (stateful !== tool) add(declare(stateful))
            }
            return AgentToolMount(declarations, stateful.defaultUserToolPhases.map { it.toString() })
        }
    }
}
