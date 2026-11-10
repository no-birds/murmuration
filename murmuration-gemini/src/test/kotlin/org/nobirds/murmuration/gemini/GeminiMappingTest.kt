package org.nobirds.murmuration.gemini

import org.nobirds.murmuration.prompt.CallIds
import com.fasterxml.jackson.databind.ObjectMapper
import com.google.genai.types.Candidate
import com.google.genai.types.Content
import com.google.genai.types.FunctionCall
import com.google.genai.types.GenerateContentResponse
import com.google.genai.types.Part
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.tooling.Instrument
import org.nobirds.murmuration.tooling.InstrumentFactory
import org.nobirds.murmuration.tooling.InstrumentParam
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.ToolCall
import org.nobirds.murmuration.prompt.ToolCallPromptPart
import org.nobirds.murmuration.prompt.ToolResultPromptPart
import org.nobirds.murmuration.prompt.toAny
import org.nobirds.murmuration.schema.SchemaInfo
import org.nobirds.murmuration.schema.AnyOfSchema
import org.nobirds.murmuration.schema.ArraySchema
import org.nobirds.murmuration.schema.IntegerSchema
import org.nobirds.murmuration.schema.NumberSchema
import org.nobirds.murmuration.schema.StringSchema
import java.util.Base64
import kotlin.jvm.optionals.getOrNull
import kotlin.reflect.full.createType
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class GeminiMappingTest {
    @Serializable
    data class TestAnnotatedModel(
        @SchemaInfo("Annotated field")
        val annotated: String,

        val notAnnotated: String
    )

    @Test
    fun testOnlyAnnotatedFieldsAreIncluded() {
        val provider = GeminiJsonSchemaProvider()
        val schema = provider.generateSchemaFromType(TestAnnotatedModel::class)
        val properties = schema.properties().getOrNull()

        assertNotNull(properties)
        assertTrue(properties!!.containsKey("annotated"), "Should contain annotated field")
        assertTrue(!properties.containsKey("notAnnotated"), "Should NOT contain non-annotated field")
    }


    @Serializable
    data class ComplexParam(
        @SchemaInfo("The id") val id: Int,
        @SchemaInfo("The name") val name: String
    )

    @Instrument
    interface TestTools {
        @Instrument("Get information about an entity by its ID and name")
        fun getInfo(
            @InstrumentParam("The unique identifier") id: Int,
            @InstrumentParam("The name of the entity") name: String,
            @InstrumentParam("Optional parameter") optional: String = "default"
        ): String

        @Instrument("Process a complex parameter")
        fun processComplex(
            @InstrumentParam("The complex object") param: ComplexParam
        ): String
    }

    @Instrument
    class ConcreteTestTools : TestTools {
        @Instrument("Get information about an entity by its ID and name")
        override fun getInfo(
            @InstrumentParam("The unique identifier") id: Int,
            @InstrumentParam("The name of the entity") name: String,
            @InstrumentParam("Optional parameter") optional: String
        ): String = ""

        @Instrument("Process a complex parameter")
        override fun processComplex(
            @InstrumentParam("The complex object") param: ComplexParam
        ): String = ""
    }

    @Test
    fun testToGeminiFunction() {
        val tools = ConcreteTestTools()
        val instruments = InstrumentFactory().create(tools).instruments
        val instrument = instruments.find { it.name == "getInfo" }
        assertNotNull(instrument)

        val function = GeminiToolDeclarations.of(listOf(instrument), GeminiWireKeyLayout.NONE).single()

        assertEquals("getInfo", function.name().getOrNull())
        assertEquals("Get information about an entity by its ID and name", function.description().getOrNull())

        val params = function.parameters().getOrNull()
        assertNotNull(params)
        assertEquals("OBJECT", params.type().getOrNull()?.toString())

        val properties = params.properties().getOrNull()
        assertNotNull(properties)
        assertTrue(properties.containsKey("id"))
        assertTrue(properties.containsKey("name"))
        assertTrue(properties.containsKey("optional"))

        assertEquals("The unique identifier", properties["id"]?.description()?.getOrNull())
        assertEquals("The name of the entity", properties["name"]?.description()?.getOrNull())

        val required = params.required().getOrNull()
        assertNotNull(required)
        assertTrue(required.contains("id"))
        assertTrue(required.contains("name"))
        assertTrue(!required.contains("optional"))
    }

    @Test
    fun testToGeminiFunctionWithComplexParam() {
        val tools = ConcreteTestTools()
        val instruments = InstrumentFactory().create(tools).instruments
        val instrument = instruments.find { it.name == "processComplex" }
        assertNotNull(instrument)

        val function = GeminiToolDeclarations.of(listOf(instrument), GeminiWireKeyLayout.NONE).single()

        assertEquals("processComplex", function.name().getOrNull())

        val params = function.parameters().getOrNull()
        assertNotNull(params)

        val properties = params.properties().getOrNull()
        assertNotNull(properties)
        assertTrue(properties.containsKey("param"))

        val paramSchema = properties["param"]
        assertNotNull(paramSchema)
        assertEquals("OBJECT", paramSchema.type().getOrNull()?.toString())
        assertEquals("The complex object", paramSchema.description().getOrNull())

        val complexProperties = paramSchema.properties().getOrNull()
        assertNotNull(complexProperties)
        assertTrue(complexProperties.containsKey("id"))
        assertTrue(complexProperties.containsKey("name"))
        assertEquals("INTEGER", complexProperties["id"]?.type()?.getOrNull()?.toString())
        assertEquals("STRING", complexProperties["name"]?.type()?.getOrNull()?.toString())
    }

    @Test
    fun testGenerateSchemaFromKType() {
        val provider = GeminiJsonSchemaProvider()

        val schemaInt = provider.generateSchemaFromKType(Int::class.createType())
        assertEquals("INTEGER", schemaInt.type().getOrNull()?.toString())

        val schemaComplex = provider.generateSchemaFromKType(ComplexParam::class.createType())
        assertEquals("OBJECT", schemaComplex.type().getOrNull()?.toString())
        assert(schemaComplex.properties().getOrNull()?.containsKey("id") == true)
        assert(schemaComplex.properties().getOrNull()?.containsKey("name") == true)
    }

    @Test
    fun `audience union maps without sibling constraints above anyOf`() {
        val mapped = AnyOfSchema(
            variants = listOf(
                StringSchema(enumValues = listOf("all", "nobody", "party")),
                ArraySchema(items = StringSchema(format = "uuid"), minItems = 1),
            ),
        ).toGeminiSchema()

        assertEquals(null, mapped.type().getOrNull())
        assertEquals(null, mapped.description().getOrNull())
        val variants = assertNotNull(mapped.anyOf().getOrNull())
        assertEquals(2, variants.size)
        assertEquals(listOf("all", "nobody", "party"), variants[0].enum_().getOrNull())
        assertEquals("ARRAY", variants[1].type().getOrNull()?.toString())
        assertEquals(1L, variants[1].minItems().getOrNull())
        val item = assertNotNull(variants[1].items().getOrNull())
        assertEquals("STRING", item.type().getOrNull()?.toString())
        assertEquals(null, item.format().getOrNull())
        assertEquals("format: uuid", item.description().getOrNull())
    }

    @Test
    fun `uuid format is demoted to description hint, date-time is forwarded`() {
        val bare = StringSchema(format = "uuid").toGeminiSchema()
        assertEquals(null, bare.format().getOrNull())
        assertEquals("format: uuid", bare.description().getOrNull())

        val described = StringSchema(format = "uuid", description = "Character id").toGeminiSchema()
        assertEquals(null, described.format().getOrNull())
        assertEquals("Character id (format: uuid)", described.description().getOrNull())

        val dateTime = StringSchema(format = "date-time", description = "When").toGeminiSchema()
        assertEquals("date-time", dateTime.format().getOrNull())
        assertEquals("When", dateTime.description().getOrNull())

        val nested = ArraySchema(items = StringSchema(format = "uuid")).toGeminiSchema()
        val item = assertNotNull(nested.items().getOrNull())
        assertEquals(null, item.format().getOrNull())
        assertEquals("format: uuid", item.description().getOrNull())
    }

    @Test
    fun `numeric formats outside the documented set are demoted`() {
        assertEquals("int64", IntegerSchema(format = "int64").toGeminiSchema().format().getOrNull())
        assertEquals("double", NumberSchema(format = "double").toGeminiSchema().format().getOrNull())
        val odd = IntegerSchema(format = "uint32").toGeminiSchema()
        assertEquals(null, odd.format().getOrNull())
        assertEquals("format: uint32", odd.description().getOrNull())
    }

    @Test
    fun testToJsonElementWithJackson() {
        val mapper = ObjectMapper()
        val node = mapper.readTree("""{"a": 1, "b": "test", "c": [true, null], "d": {"e": 1.5}}""")

        val json = node.toJsonElement()
        assertTrue(json is JsonObject)
        assertEquals(JsonPrimitive(1), json["a"])
        assertEquals(JsonPrimitive("test"), json["b"])

        val array = json["c"]
        assertTrue(array is JsonArray)
        assertEquals(JsonPrimitive(true), array[0])
        assertEquals(JsonNull, array[1])

        val nested = json["d"]
        assertTrue(nested is JsonObject)
        assertEquals(JsonPrimitive(1.5), nested["e"])
    }

    @Test
    fun testToAny() {
        val json = JsonObject(mapOf(
            "a" to JsonPrimitive(1),
            "b" to JsonPrimitive("test"),
            "c" to JsonArray(listOf(JsonPrimitive(true), JsonNull)),
            "d" to JsonObject(mapOf("e" to JsonPrimitive(1.5)))
        ))

        val any = json.toAny()
        assertTrue(any is Map<*, *>)
        assertEquals(1L, any["a"])
        assertEquals("test", any["b"])

        val list = any["c"]
        assertTrue(list is List<*>)
        assertEquals(true, list[0])
        assertEquals(null, list[1])

        val nested = any["d"]
        assertTrue(nested is Map<*, *>)
        assertEquals(1.5, nested["e"])
    }

    @Test
    fun testPromptPartToGenai() {
        val call = ToolCall("test_tool", mapOf("arg1" to JsonPrimitive("val1")), callId = CallIds.mint())
        val callPart = ToolCallPromptPart(call)
        val genaiCallPart = GeminiHistoryConverter.EMPTY.toGenai(callPart)

        assertNotNull(genaiCallPart.functionCall().getOrNull())
        assertEquals("test_tool", genaiCallPart.functionCall().getOrNull()?.name()?.getOrNull())
        assertEquals("val1", genaiCallPart.functionCall().getOrNull()?.args()?.getOrNull()?.get("arg1"))

        val resultPart = ToolResultPromptPart("test_tool", JsonObject(mapOf("result" to JsonPrimitive("ok"))), callId = CallIds.mint())
        val genaiResultPart = GeminiHistoryConverter.EMPTY.toGenai(resultPart)

        assertNotNull(genaiResultPart.functionResponse().getOrNull())
        assertEquals("test_tool", genaiResultPart.functionResponse().getOrNull()?.name()?.getOrNull())
        assertEquals("ok", (genaiResultPart.functionResponse().getOrNull()?.response()?.getOrNull()?.get("result")))
    }

    @Test
    fun testThoughtSignatureMapping() {
        val signature = "test-signature".toByteArray()
        val signatureBase64 = Base64.getEncoder().encodeToString(signature)

        val part = Part.builder()
            .functionCall(FunctionCall.builder().name("test").args(mapOf()).build())
            .thoughtSignature(signature)
            .build()

        val response = GenerateContentResponse.builder()
            .candidates(listOf(Candidate.builder().content(Content.builder().parts(listOf(part)).build()).build()))
            .build()

        val events = response.allParts().mapNotNull { it.toEventOrNull() }
        val toolCallParts = events.filterIsInstance<PromptExecutionEvent.PartReceived>()
            .mapNotNull { it.part as? MessagePart.ToolCall }
        assertEquals(1, toolCallParts.size)
        assertEquals(signatureBase64, toolCallParts[0].thoughtSignature)

        // Test back to Genai
        val toolCallPart = toolCallParts[0]
        val toolCall = ToolCall(toolCallPart.name, toolCallPart.args, toolCallPart.thoughtSignature, callId = CallIds.mint())
        val callPart = ToolCallPromptPart(toolCall)
        val genaiPart = GeminiHistoryConverter.EMPTY.toGenai(callPart)

        assertArrayEquals(signature, genaiPart.thoughtSignature().getOrNull())
    }

    @Test
    fun testToolResultThoughtSignatureMapping() {
        val signature = "result-signature".toByteArray()
        val signatureBase64 = Base64.getEncoder().encodeToString(signature)

        val resultPart = ToolResultPromptPart("test", JsonPrimitive("ok"), CallIds.mint(), signatureBase64)
        val genaiPart = GeminiHistoryConverter.EMPTY.toGenai(resultPart)

        assertArrayEquals(signature, genaiPart.thoughtSignature().getOrNull())
    }

    @Test
    fun testToolContentMapsToUserRoleForGenai() {
        // Vertex's live generateContent path only accepts role user/model — role="function"
        // triggers `400 Please use a valid role: user, model`. Regression test for that bug.
        val toolContent = PromptContent(
            type = PromptContentType.TOOL,
            parts = listOf(ToolResultPromptPart("test_tool", JsonPrimitive("ok"), callId = CallIds.mint())),
        )

        val genaiContents = GeminiHistoryConverter.EMPTY.toGenaiContents(listOf(toolContent))

        assertEquals(1, genaiContents.size)
        assertEquals("user", genaiContents[0].role().getOrNull())
        assertTrue(genaiContents[0].role().getOrNull() != "function")
    }

    @Test
    fun testFunctionRoleReadsBackAsToolContentType() {
        // Legacy dumps written before the toGenaiContents fix used role="function" for tool
        // exchanges. The reverse mapper must still classify those as TOOL, not fall through
        // to the USER default.
        val content = Content.builder()
            .role("function")
            .parts(listOf(Part.builder().text("legacy tool result").build()))
            .build()

        val message = content.toMessage()

        assertEquals(PromptContentType.TOOL, message.role)
    }
}
