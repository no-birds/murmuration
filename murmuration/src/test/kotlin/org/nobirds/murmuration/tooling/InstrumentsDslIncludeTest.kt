package org.nobirds.murmuration.tooling

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Whitelist form of the group filter (`from(donor) { include(...) }`): keep exactly the named
 * tools, fail loudly on a name the donor does not declare.
 *
 * The failure case is the point of the feature: expressing a narrow surface by subtraction
 * (`exclude(everything - keep)`) cannot notice a renamed or removed tool, so a stale whitelist
 * used to widen silently. Here it stops the build.
 */
class InstrumentsDslIncludeTest {

    @Suppress("unused")
    class MemoryDonor {
        @Instrument("Add a memory record.")
        fun addMemory(@InstrumentParam("Text") text: String): String = "added $text"

        @Instrument("Update a memory record.")
        fun updateMemory(@InstrumentParam("Text") text: String): String = "updated $text"

        @Instrument("Rewrite the whole character sheet.")
        fun updateSheet(@InstrumentParam("Text") text: String): String = "rewritten $text"

        @Instrument("Move to another phase.")
        fun transitionTo(@InstrumentParam("Phase") phase: String): String = "moved to $phase"
    }

    private enum class DemoState { WORKING }

    @Test
    fun `include keeps exactly the named tools`() {
        val tools = buildInstruments {
            from(MemoryDonor()) { include("addMemory", "updateMemory") }
        }
        assertEquals(listOf("addMemory", "updateMemory"), tools.map { it.name })
    }

    @Test
    fun `only is the same filter under a call-site-friendly name`() {
        val tools = buildInstruments {
            from(MemoryDonor()) { only("updateSheet") }
        }
        assertEquals(listOf("updateSheet"), tools.map { it.name })
    }

    @Test
    fun `include composes with inPhases like exclude does`() {
        val tools = buildInstruments {
            from(MemoryDonor()) { include("addMemory"); inPhases(DemoState.WORKING) }
        }
        val declaration = tools.single()
        assertEquals("addMemory", declaration.name)
        assertTrue(
            declaration.annotations.filterIsInstance<Phases>().single().phases.contains("WORKING"),
            "phases tagging must not depend on which filter form the group used",
        )
    }

    @Test
    fun `unknown name in include is a build error, not a silent skip`() {
        val failure = assertFailsWith<IllegalArgumentException> {
            buildInstruments {
                from(MemoryDonor()) { include("addMemory", "addMemoryV2") }
            }
        }
        assertTrue("addMemoryV2" in failure.message.orEmpty(), "the stale name is named: ${failure.message}")
        assertTrue("MemoryDonor" in failure.message.orEmpty(), "the donor is named: ${failure.message}")
    }

    @Test
    fun `empty include is a build error, not an empty schema`() {
        // `include()` with no names used to hand the agent an EMPTY tool set — a tool-less agent
        // that silently does nothing (no crash, no tools, one wasted paid run to notice).
        val failure = assertFailsWith<IllegalArgumentException> {
            buildInstruments {
                from(MemoryDonor()) { include() }
            }
        }
        assertTrue("no names" in failure.message.orEmpty(), failure.message.orEmpty())

        val aliasFailure = assertFailsWith<IllegalArgumentException> {
            buildInstruments {
                from(MemoryDonor()) { only() }
            }
        }
        assertTrue("no names" in aliasFailure.message.orEmpty(), aliasFailure.message.orEmpty())
    }

    @Test
    fun `include and exclude in one group is a build error`() {
        val failure = assertFailsWith<IllegalArgumentException> {
            buildInstruments {
                from(MemoryDonor()) { include("addMemory"); exclude("transitionTo") }
            }
        }
        assertTrue("pick one" in failure.message.orEmpty(), failure.message.orEmpty())
    }

    @Test
    fun `exclude keeps working untouched`() {
        val tools = buildInstruments {
            from(MemoryDonor()) { exclude("transitionTo", "updateSheet") }
        }
        assertEquals(listOf("addMemory", "updateMemory"), tools.map { it.name })
    }
}
