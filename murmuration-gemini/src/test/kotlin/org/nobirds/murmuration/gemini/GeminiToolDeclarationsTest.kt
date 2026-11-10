package org.nobirds.murmuration.gemini

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.tooling.InstrumentFactory
import kotlin.jvm.optionals.getOrNull

/** Canonical sample: both the request and cache-create paths consume [GeminiToolDeclarations] only. */
class GeminiToolDeclarationsTest {
    private val instruments = InstrumentFactory().create(GeminiMappingTest.ConcreteTestTools()).instruments

    @Test
    fun `tool carries exactly the declarations of the instruments, in order`() {
        val declarations = GeminiToolDeclarations.of(instruments, GeminiWireKeyLayout.NONE)
        val tool = GeminiToolDeclarations.toolOf(instruments, GeminiWireKeyLayout.NONE)

        assertEquals(instruments.map { it.name }, declarations.map { it.name().getOrNull() })
        assertEquals(declarations, tool?.functionDeclarations()?.getOrNull())
    }

    @Test
    fun `mapping is deterministic across calls`() {
        assertEquals(GeminiToolDeclarations.of(instruments, GeminiWireKeyLayout.NONE), GeminiToolDeclarations.of(instruments, GeminiWireKeyLayout.NONE))
    }

    @Test
    fun `no instruments - no tool`() {
        assertNull(GeminiToolDeclarations.toolOf(emptyList(), GeminiWireKeyLayout.NONE))
    }
}
