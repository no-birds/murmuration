package org.nobirds.murmuration.gemini

import org.nobirds.murmuration.prompt.CallIds
import com.google.genai.types.Schema
import com.google.genai.types.Type
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.gemini.GeminiToolDeclarations.rewrite
import org.nobirds.murmuration.gemini.GeminiToolDeclarations.wireKeyMapOf
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.ToolCall
import org.nobirds.murmuration.prompt.ToolCallPromptPart
import org.nobirds.murmuration.schema.SchemaInfo
import org.nobirds.murmuration.schema.toFunctionSchema
import org.nobirds.murmuration.tooling.Instrument
import org.nobirds.murmuration.tooling.InstrumentParam
import kotlin.jvm.optionals.getOrNull
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class GeminiWireLayoutTest {

    @Serializable data class Inner(@SchemaInfo("i") val id: Int, @SchemaInfo("n") val name: String)

    @Instrument
    class Tools {
        @Instrument("d")
        fun act(
            @InstrumentParam("p") p: Inner,
            @InstrumentParam("list") list: List<Inner>,
            @InstrumentParam("opt") opt: String? = null,
        ): String = ""
    }

    private val instruments = WireLayoutTestSupport.instrumentsOf(Tools())
    private val schemas = WireLayoutTestSupport.canonicalSchemas(instruments)
    private val testLayout = WireLayoutTestSupport.layoutOf(schemas)

    private fun obj(vararg p: Pair<String, JsonElement>) = JsonObject(mapOf(*p))

    private val canonicalArgs: Map<String, JsonElement> = mapOf(
        "p" to obj("id" to JsonPrimitive(1), "name" to JsonPrimitive("a")),
        "list" to JsonArray(listOf(obj("id" to JsonPrimitive(2), "name" to JsonPrimitive("b")))),
        "opt" to JsonPrimitive("x"),
    )

    private fun keysOf(schema: Schema) = schema.properties().orElse(emptyMap()).keys

    @Test
    fun `round trip toCanonical of toWire is identity on a non-empty table`() {
        val built = GeminiToolDeclarations.build(instruments, testLayout)
        val map = built.wireKeyMaps.getValue("act")
        val wire = map.toWire(canonicalArgs)

        assertNotEquals(canonicalArgs, wire)
        assertTrue(wire.keys.all { it.startsWith("~") }, "all top-level keys moved: ${wire.keys}")
        assertEquals(JsonObject(canonicalArgs), JsonObject(wire).toCanonical())
        assertEquals(canonicalArgs, wire.toCanonicalArgs())
    }

    @Test
    fun `toWire yields exactly the keys the wire declaration declares`() {
        val built = GeminiToolDeclarations.build(instruments, testLayout)
        val declared = built.declarations.single().parameters().get()
        val wire = built.wireKeyMaps.getValue("act").toWire(canonicalArgs)

        assertEquals(keysOf(declared), wire.keys)
        val wireInner = declared.properties().get().entries.single { it.value.type().getOrNull()?.knownEnum() == Type.Known.OBJECT }.value
        val wireP = wire.entries.single { (it.value as? JsonObject)?.size == 2 }.value as JsonObject
        assertEquals(keysOf(wireInner), wireP.keys)
        assertTrue(declared.required().get().all { it in declared.properties().get() })
    }

    @Test
    fun `empty table or NONE gives byte-identical declaration and no wire maps`() {
        val old = instruments.map { instrument ->
            val schema = instrument.toFunctionSchema()
            com.google.genai.types.FunctionDeclaration.builder()
                .name(schema.name).description(schema.description)
                .parameters(schema.parameters.toGeminiSchema()).build()
        }
        val oldToolJson = com.google.genai.types.Tool.builder().functionDeclarations(old).build().toJson()
        for (layout in listOf(GeminiWireKeyLayout.NONE, GeminiWireKeyLayout.ofSignatures(emptyMap()))) {
            val built = GeminiToolDeclarations.build(instruments, layout)
            assertEquals(old, built.declarations)
            assertEquals(oldToolJson, built.tool!!.toJson())
            assertTrue(built.wireKeyMaps.isEmpty())
        }
    }

    @Test
    fun `history conversion with NONE equals the plain conversion`() {
        val content = PromptContent(PromptContentType.MODEL, listOf(ToolCallPromptPart(ToolCall("act", canonicalArgs, callId = CallIds.mint()))))
        val none = GeminiToolDeclarations.build(instruments, GeminiWireKeyLayout.NONE).historyConverter
        assertEquals(GeminiHistoryConverter.EMPTY.toGenaiContents(listOf(content)), none.toGenaiContents(listOf(content)))
    }

    @Test
    fun `history conversion renames only by the map and leaves unmounted tools canonical`() {
        val built = GeminiToolDeclarations.build(instruments, testLayout)
        val mounted = PromptContent(PromptContentType.MODEL, listOf(ToolCallPromptPart(ToolCall("act", canonicalArgs, callId = CallIds.mint()))))
        val foreign = PromptContent(PromptContentType.MODEL, listOf(ToolCallPromptPart(ToolCall("other", canonicalArgs, callId = CallIds.mint()))))

        val keys = { c: PromptContent ->
            built.historyConverter.toGenaiContents(listOf(c)).single().parts().get().single()
                .functionCall().get().args().get().keys
        }
        assertTrue(keys(mounted).all { it.startsWith("~") })
        assertEquals(canonicalArgs.keys, keys(foreign))
        assertTrue(TextPromptPart("t").let { built.historyConverter.toGenai(it).text().isPresent })
    }

    private fun objectSchema(vararg fields: Pair<String, Schema>, required: List<String> = emptyList()): Schema =
        Schema.builder().type(Type.Known.OBJECT).properties(mapOf(*fields)).required(required).build()

    private fun scalar(type: Type.Known): Schema = Schema.builder().type(type).build()

    @Test
    fun `anyOf group - a key gets a wire name only if all variants that declare it agree`() {
        val v1 = objectSchema("a" to scalar(Type.Known.STRING), "b" to scalar(Type.Known.INTEGER), "only1" to scalar(Type.Known.BOOLEAN))
        val v2 = objectSchema("a" to scalar(Type.Known.STRING), "b" to scalar(Type.Known.STRING))
        val root = objectSchema(
            "grp" to Schema.builder().anyOf(listOf(v1, v2)).build(),
            "top" to scalar(Type.Known.STRING),
        )
        val layout = WireLayoutTestSupport.layoutOf(listOf(root))
        val map = wireKeyMapOf(root, layout)

        assertTrue(map.wireNameAt(listOf("grp", "a"))!!.endsWith("a"), "a agrees in both variants -> renamed")
        assertEquals(null, map.wireNameAt(listOf("grp", "b")), "b differs between variants -> canonical in all")
        assertTrue(map.wireNameAt(listOf("grp", "only1")) != null, "declared in one variant only -> renamed")
        assertTrue(map.wireNameAt(listOf("top")) != null)

        val wire = root.rewrite(emptyList(), map)
        val wireVariants = wire.properties().get().getValue(map.wireNameAt(listOf("grp")) ?: "grp").anyOf().get()
        assertTrue("b" in keysOf(wireVariants[0]) && "b" in keysOf(wireVariants[1]))
        assertTrue(keysOf(wireVariants[0]).none { it == "a" } && keysOf(wireVariants[1]).none { it == "a" })
        assertEquals(keysOf(wireVariants[0]).first(), map.wireNameAt(listOf("grp", "a")))
        assertEquals(keysOf(wireVariants[1]).first(), map.wireNameAt(listOf("grp", "a")))
    }

    @Test
    fun `a canonical key starting with the marker is rejected`() {
        val root = objectSchema("~bad" to scalar(Type.Known.STRING))
        assertFailsWith<IllegalArgumentException> { wireKeyMapOf(root, WireLayoutTestSupport.layoutOf(listOf(root))) }
    }
}
