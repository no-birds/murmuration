package org.nobirds.murmuration.tooling

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.InstrumentDeclaration
import org.nobirds.murmuration.prompt.InstrumentParameter
import org.nobirds.murmuration.prompt.toAny
import org.nobirds.murmuration.schema.toFunctionSchema
import java.util.UUID
import kotlin.reflect.KType
import kotlin.reflect.typeOf
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Instrument wire arguments (design I8): the schema and the coercion of `java.util.UUID`
 * parameters at any nesting level, and [InstrumentDeclaration.prepareWireArguments] as an
 * interface member rather than an annotation flag.
 *
 * Canonical sample for the tooling: declare a reflective tool, take its
 * [toFunctionSchema] for the schema side, and feed raw provider JSON through
 * `prepareWireArguments` -> `invoke` for the call side.
 */
class InstrumentWireArgumentsTest {

    @Suppress("unused")
    class UuidTools {
        var calls = 0

        @Instrument("Captures uuid parameters of every shape")
        fun capture(
            @InstrumentParam("Required id") id: UUID,
            @InstrumentParam("Optional id") optionalId: UUID? = null,
            @InstrumentParam("Id list") ids: List<UUID>,
            @InstrumentParam("Id set") idSet: Set<UUID>? = null,
            @InstrumentParam("Nested") nested: List<List<UUID>?>? = null,
        ): String {
            calls += 1
            return "ok"
        }
    }

    @Suppress("unused")
    class CapturingTools {
        var captured: List<Any?> = emptyList()

        @Instrument("Echo")
        fun echo(
            @InstrumentParam("Required id") id: UUID,
            @InstrumentParam("Optional id") optionalId: UUID? = null,
            @InstrumentParam("Id list") ids: List<UUID>,
            @InstrumentParam("Id set") idSet: Set<UUID>? = null,
            @InstrumentParam("Nested") nested: Map<String, List<UUID?>>? = null,
        ): String {
            captured = listOf(id, optionalId, ids, idSet, nested)
            return "ok"
        }
    }

    @Suppress("unused")
    class JsonTools {
        @Instrument("Raw json")
        fun raw(
            @InstrumentParam("Payload") payload: JsonElement,
            @InstrumentParam("Count") count: Int,
        ): Pair<JsonElement, Int> = payload to count
    }

    class CustomJsonTool : InstrumentDeclaration<Any?> {
        override val obj: Any get() = this
        override val name: String = "customJson"
        override val description: String = "Custom JSON tool"
        override val parameters = listOf(InstrumentParameter("payload", typeOf<JsonElement>()))
        override val returnType: KType = typeOf<Any?>()
        override suspend fun invoke(args: Map<String, Any?>): Any? = args["payload"]
    }

    private fun JsonObject.uuidString(): Boolean =
        this["type"]?.jsonPrimitive?.content == "string" && this["format"]?.jsonPrimitive?.content == "uuid"

    @Test
    fun `schema renders UUID as string with uuid format at every level`() {
        val schema = InstrumentFactory().create(UuidTools()).instruments.single().toFunctionSchema()
        val json = schema.parameters.toJson()
        val props = json["properties"]!!.jsonObject

        assertTrue(props["id"]!!.jsonObject.uuidString())

        val optional = props["optionalId"]!!.jsonObject
        assertTrue(optional.uuidString())
        assertEquals(true, optional["nullable"]!!.jsonPrimitive.content.toBoolean())

        val ids = props["ids"]!!.jsonObject
        assertEquals("array", ids["type"]!!.jsonPrimitive.content)
        assertTrue(ids["items"]!!.jsonObject.uuidString())

        val idSet = props["idSet"]!!.jsonObject
        assertEquals("array", idSet["type"]!!.jsonPrimitive.content)
        assertTrue(idSet["items"]!!.jsonObject.uuidString())

        val nested = props["nested"]!!.jsonObject
        assertEquals("array", nested["type"]!!.jsonPrimitive.content)
        val inner = nested["items"]!!.jsonObject
        assertEquals("array", inner["type"]!!.jsonPrimitive.content)
        assertTrue(inner["items"]!!.jsonObject.uuidString())

        val required = json["required"]!!.jsonArray.map { it.jsonPrimitive.content }.toSet()
        assertEquals(setOf("id", "ids"), required)
    }

    @Test
    fun `wire strings coerce to UUID UUID null List and Set`() = runBlocking {
        val tools = CapturingTools()
        val echo = InstrumentFactory().create(tools).instruments.single()
        val id = UUID.randomUUID()
        val optional = UUID.randomUUID()
        val a = UUID.randomUUID()
        val b = UUID.randomUUID()
        val wire = buildJsonObject {
            put("id", id.toString())
            put("optionalId", optional.toString())
            put("ids", buildJsonArray { add(JsonPrimitive(a.toString())); add(JsonPrimitive(b.toString())) })
            put("idSet", buildJsonArray { add(JsonPrimitive(a.toString())); add(JsonPrimitive(a.toString())) })
            put("nested", buildJsonObject {
                put("k", buildJsonArray { add(JsonPrimitive(b.toString())); add(JsonNull) })
            })
        }

        assertEquals("ok", echo.invoke(echo.prepareWireArguments(wire)))
        assertEquals(
            listOf(id, optional, listOf(a, b), setOf(a), mapOf("k" to listOf(b, null))),
            tools.captured,
        )
    }

    @Test
    fun `absent and null optional UUID arrive as null`() = runBlocking {
        val tools = CapturingTools()
        val echo = InstrumentFactory().create(tools).instruments.single()
        val id = UUID.randomUUID()

        echo.invoke(echo.prepareWireArguments(buildJsonObject {
            put("id", id.toString())
            put("ids", buildJsonArray { })
        }))
        assertEquals(listOf(id, null, emptyList<UUID>(), null, null), tools.captured)

        echo.invoke(echo.prepareWireArguments(buildJsonObject {
            put("id", id.toString())
            put("optionalId", JsonNull)
            put("ids", buildJsonArray { })
        }))
        assertEquals(listOf(id, null, emptyList<UUID>(), null, null), tools.captured)
    }

    @Test
    fun `invalid UUID inside a list fails before the tool body`() = runBlocking {
        val tools = UuidTools()
        val capture = InstrumentFactory().create(tools).instruments.single()
        val wire = buildJsonObject {
            put("id", UUID.randomUUID().toString())
            put("ids", buildJsonArray { add(JsonPrimitive("not-a-uuid")) })
        }

        assertFailsWith<IllegalArgumentException> { capture.invoke(capture.prepareWireArguments(wire)) }
        assertEquals(0, tools.calls)
    }

    @Test
    fun `reflective declaration passes JsonElement parameters raw and converts the rest`() = runBlocking {
        val raw = InstrumentFactory().create(JsonTools()).instruments.single()
        val payloads = listOf(
            JsonPrimitive(true),
            Json.parseToJsonElement("9223372036854775808"),
            JsonNull,
            buildJsonObject { put("visibility", "all") },
            buildJsonArray { add(JsonPrimitive("all")); add(JsonPrimitive(2)) },
        )

        for (payload in payloads) {
            val prepared = raw.prepareWireArguments(mapOf("payload" to payload, "count" to JsonPrimitive(3)))
            assertEquals(payload, prepared["payload"])
            assertEquals(payload to 3, raw.invoke(prepared))
        }
    }

    @Test
    fun `custom declaration keeps the default toAny conversion`() = runBlocking {
        val tool = CustomJsonTool()
        val payload = Json.parseToJsonElement("9223372036854775808")
        val prepared = tool.prepareWireArguments(mapOf("payload" to payload))

        assertEquals(payload.toAny(), prepared["payload"])
        assertEquals(payload.toAny(), tool.invoke(prepared))
    }

    @Test
    fun `decorators and wrappers forward prepareWireArguments to the reflective declaration`() {
        val payload = buildJsonObject { put("visibility", "all") }
        val wire = mapOf("payload" to payload, "label" to JsonPrimitive("x"))

        // Decorated by the factory chain (a parameter validator wraps the declaration).
        val decorated = InstrumentFactory().create(ValidatedJsonTools()).instruments.single()
        assertTrue(decorated !is KFunctionInstrumentDeclaration<*>, "factory must have wrapped the declaration")
        assertEquals(payload, decorated.prepareWireArguments(wire)["payload"])

        // Wrapped by an annotation-adding wrapper (phase defaults).
        val phased = listOf(InstrumentFactory().create(JsonTools()).instruments.single())
            .withDefaultPhases(listOf("WORKING"))
            .single()
        assertTrue(phased !is KFunctionInstrumentDeclaration<*>, "phase defaults must have wrapped the declaration")
        assertEquals(payload, phased.prepareWireArguments(wire)["payload"])
        assertNull(phased.prepareWireArguments(mapOf("x" to JsonNull))["x"])
    }

    @Suppress("unused")
    class ValidatedJsonTools {
        @Instrument("Validated raw json")
        fun raw(
            @InstrumentParam("Payload") payload: JsonElement,
            @NotBlank @InstrumentParam("Label") label: String = "x",
        ): String = "ok"
    }
}
