package org.nobirds.murmuration.gemini

import com.google.genai.Client
import kotlinx.serialization.Serializable
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.InstrumentsDeclaration
import org.nobirds.murmuration.prompt.ToolsPromptSettings
import org.nobirds.murmuration.schema.SchemaInfo
import org.nobirds.murmuration.tooling.Instrument
import org.nobirds.murmuration.tooling.InstrumentFactory
import org.nobirds.murmuration.tooling.InstrumentParam

/** The explicit-cache key must see the whole wire declaration, not just top-level parameters. */
class GeminiCacheHashTest {
    private val backend = GeminiCacheBackend(Client.builder().apiKey("test").build(), GeminiWireKeyLayout.NONE)

    enum class Color { RED, GREEN }
    enum class ColorMore { RED, GREEN, BLUE }

    @Serializable data class Base(@SchemaInfo("x") val x: Int, @SchemaInfo("c") val c: Color)
    @Serializable data class RenamedField(@SchemaInfo("x") val y: Int, @SchemaInfo("c") val c: Color)
    @Serializable data class ChangedType(@SchemaInfo("x") val x: String, @SchemaInfo("c") val c: Color)
    @Serializable data class ChangedEnum(@SchemaInfo("x") val x: Int, @SchemaInfo("c") val c: ColorMore)
    @Serializable data class ChangedDescription(@SchemaInfo("x!") val x: Int, @SchemaInfo("c") val c: Color)
    @Serializable data class OptionalField(@SchemaInfo("x") val x: Int, @SchemaInfo("c") val c: Color? = null)

    @Instrument class ToolBase { @Instrument("d") fun act(@InstrumentParam("p") p: Base): String = "" }
    @Instrument class ToolRenamedField { @Instrument("d") fun act(@InstrumentParam("p") p: RenamedField): String = "" }
    @Instrument class ToolChangedType { @Instrument("d") fun act(@InstrumentParam("p") p: ChangedType): String = "" }
    @Instrument class ToolChangedEnum { @Instrument("d") fun act(@InstrumentParam("p") p: ChangedEnum): String = "" }
    @Instrument class ToolChangedDescription { @Instrument("d") fun act(@InstrumentParam("p") p: ChangedDescription): String = "" }
    @Instrument class ToolOptionalField { @Instrument("d") fun act(@InstrumentParam("p") p: OptionalField): String = "" }
    @Instrument class ToolRenamedTool { @Instrument("d") fun act2(@InstrumentParam("p") p: Base): String = "" }
    @Instrument class ToolOptionalParam { @Instrument("d") fun act(@InstrumentParam("p") p: Base? = null): String = "" }

    private fun hashOf(tools: Any): String {
        val instruments = InstrumentFactory().create(tools).instruments
        return backend.hashContent(emptyList(), ToolsPromptSettings(listOf(InstrumentsDeclaration.of(tools, instruments))))
    }

    @Test
    fun `same declaration gives same hash`() {
        assertEquals(hashOf(ToolBase()), hashOf(ToolBase()))
    }

    @Test
    fun `any change of the declaration changes the hash`() {
        val base = hashOf(ToolBase())
        val variants = mapOf(
            "nested field name" to ToolRenamedField(),
            "nested field type" to ToolChangedType(),
            "enum values" to ToolChangedEnum(),
            "nested description" to ToolChangedDescription(),
            "nested required" to ToolOptionalField(),
            "top-level required" to ToolOptionalParam(),
            "tool name" to ToolRenamedTool(),
        )
        variants.forEach { (what, tools) -> assertNotEquals(base, hashOf(tools), "hash must change on: $what") }
    }
}
