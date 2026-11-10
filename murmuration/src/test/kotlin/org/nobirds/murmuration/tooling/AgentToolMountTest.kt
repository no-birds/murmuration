package org.nobirds.murmuration.tooling

import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.InstrumentsDeclaration
import org.nobirds.murmuration.prompt.ToolsPromptSettings
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * [AgentToolMount] is the single writer of "what is mounted on a tool-calling agent".
 *
 * Canonical sample: build a mount from `(tool, stateful)`, read [AgentToolMount.instruments] for the
 * mounted set and [AgentToolMount.toolsSettings] for exactly what reaches the provider.
 */
class AgentToolMountTest {

    @Suppress("unused")
    class Tools {
        @Instrument("Plain tool")
        fun plain(@InstrumentParam("Any") value: String): String = value
    }

    /** A stateful distinct from the tool: contributes its own instruments (incl. `transitionTo`). */
    @Suppress("unused")
    class Machine : Stateful<DefaultAgentState, Machine> {
        override val stateMachine = defaultStateMachine<Machine>()
        override val currentState: DefaultAgentState get() = DefaultAgentState.WORKING
        override val defaultUserToolPhases: List<DefaultAgentState> get() = listOf(DefaultAgentState.WORKING)

        @Instrument("Stateful tool")
        fun mutate(@InstrumentParam("Any") value: String): String = value

        override suspend fun transitionTo(nextState: DefaultAgentState): TransitionResult =
            TransitionResult.Success("ok")
    }

    private fun names(declarations: List<InstrumentsDeclaration<*>>) =
        declarations.flatMap { it.instruments }.map { it.name }

    @Test
    fun `mount is the union of tool and stateful declarations`() {
        val tools = Tools()
        val machine = Machine()
        val mount = AgentToolMount.of(tools, machine)

        val expected = InstrumentFactory().create(tools).instruments.map { it.name } +
            InstrumentFactory().create(machine).instruments.map { it.name }
        assertEquals(expected, names(mount.declarations))
        assertEquals(expected, mount.instruments.map { it.name })
        assertTrue("plain" in expected && "mutate" in expected && "transitionTo" in expected)
    }

    @Test
    fun `stateful that is the tool itself is mounted once`() {
        val machine = Machine()
        val mount = AgentToolMount.of(machine, machine)

        assertEquals(1, mount.declarations.size)
        assertEquals(InstrumentFactory().create(machine).instruments.map { it.name }, mount.instruments.map { it.name })
    }

    @Test
    fun `statefulOf picks the tool itself when it is Stateful`() {
        val machine = Machine()

        assertTrue(AgentToolMount.statefulOf(machine) === machine)
    }

    @Test
    fun `statefulOf falls back to defaultStateful for a plain tool`() {
        val tools = Tools()
        val stateful = AgentToolMount.statefulOf(tools)

        assertTrue(stateful as Any !== tools)
        assertEquals(DefaultAgentState.WORKING, stateful.currentState)
        val mount = AgentToolMount.of(tools, stateful)
        assertEquals(2, mount.declarations.size, "plain tool is mounted with the default stateful's declaration")
        assertTrue("transitionTo" in mount.instruments.map { it.name })
    }

    @Test
    fun `toolsSettings replaces tools and keeps other settings`() {
        val mount = AgentToolMount.of(Tools(), Machine())
        val base = ToolsPromptSettings(emptyList(), includeThoughts = false, thinkingBudget = 7)

        val first = mount.toolsSettings(base)
        val second = mount.toolsSettings(first.copy(tools = emptyList()))

        assertEquals(mount.declarations, first.tools)
        assertEquals(false, first.includeThoughts)
        assertEquals(7, first.thinkingBudget)
        assertEquals(mount.declarations, second.tools, "applying twice must not duplicate tools")
        assertEquals(mount.declarations, mount.toolsSettings(null).tools)
    }

    @Test
    fun `generator-declared tools are rejected instead of appended`() {
        val mount = AgentToolMount.of(Tools(), Machine())
        val foreign = ToolsPromptSettings(mount.declarations)

        assertFailsWith<IllegalArgumentException> { mount.toolsSettings(foreign) }
    }

    @Test
    fun `default phases are applied to mounted instruments`() {
        val mount = AgentToolMount.of(Tools(), Machine())

        val plain = mount.instruments.single { it.name == "plain" }
        assertEquals(listOf("WORKING"), plain.phases)
    }
}
