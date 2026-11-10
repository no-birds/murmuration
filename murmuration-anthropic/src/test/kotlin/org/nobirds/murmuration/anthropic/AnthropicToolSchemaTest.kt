package org.nobirds.murmuration.anthropic

import com.anthropic.core.jsonMapper
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.InvalidRequestException
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.TypedPromptSettings
import org.nobirds.murmuration.schema.FunctionSchema
import org.nobirds.murmuration.schema.SchemaInfo
import org.nobirds.murmuration.schema.jsonSchema
import org.nobirds.murmuration.schema.JsonSchemaBuilder.Kind
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class AnthropicToolSchemaTest {

    private val prompt = Prompt(AnthropicLlms.findModel("anthropic", "claude-sonnet-5-5"), null, emptyList())

    private fun wire(value: Any): JsonObject = Json.parseToJsonElement(jsonMapper().writeValueAsString(value)).jsonObject

    private fun parse(json: String): JsonObject = Json.parseToJsonElement(json).jsonObject

    /** Typical tool: nested object, nullable field, enum, array of objects. */
    private val typicalTool = FunctionSchema(
        name = "move_party",
        description = "Moves the party",
        parameters = jsonSchema {
            type(Kind.OBJECT)
            property("destination", jsonSchema {
                type(Kind.OBJECT)
                property("name", jsonSchema { type(Kind.STRING) })
                property("note", jsonSchema { type(Kind.STRING); nullable(true) })
                required("name")
            })
            property("pace", jsonSchema { type(Kind.STRING); enum("slow", "fast") })
            property("members", jsonSchema {
                type(Kind.ARRAY)
                items { type(Kind.STRING) }
            })
            required("destination", "pace")
        },
    )

    @Test
    fun `typical tool becomes input_schema with standard nullable and enum`() {
        val tool = wire(typicalTool.toAnthropicTool(prompt))

        assertEquals("move_party", tool["name"].toString().trim('"'))
        assertEquals("Moves the party", tool["description"].toString().trim('"'))
        assertEquals(
            parse(
                """{"type":"object","properties":{
                    "destination":{"type":"object","properties":{"name":{"type":"string"},"note":{"type":["string","null"]}},"required":["name"]},
                    "pace":{"type":"string","enum":["slow","fast"]},
                    "members":{"type":"array","items":{"type":"string"}}},
                  "required":["destination","pace"]}"""
            ),
            tool["input_schema"]!!.jsonObject,
        )
        assertNull(tool["strict"])
    }

    @Test
    fun `non-strict tool keeps constraints the strict subset forbids`() {
        val tool = FunctionSchema("t", "d", jsonSchema {
            type(Kind.OBJECT)
            property("n", jsonSchema { type(Kind.INTEGER); minimum(1L) })
        })

        val props = wire(tool.toAnthropicTool(prompt))["input_schema"]!!.jsonObject["properties"]!!.jsonObject
        assertEquals(parse("""{"type":"integer","minimum":1}"""), props["n"]!!.jsonObject)
    }

    @Test
    fun `strict tool closes objects and is marked strict`() {
        val tool = wire(typicalTool.toAnthropicTool(prompt, strict = true))

        assertEquals("true", tool["strict"].toString())
        val schema = tool["input_schema"]!!.jsonObject
        assertEquals("false", schema["additionalProperties"].toString())
        assertEquals("false", schema["properties"]!!.jsonObject["destination"]!!.jsonObject["additionalProperties"].toString())
    }

    @Test
    fun `every construct outside the subset is rejected before the network`() {
        val cases = mapOf(
            "minimum" to jsonSchema { type(Kind.INTEGER); minimum(1L) },
            "maximum" to jsonSchema { type(Kind.NUMBER); maximum(2.0) },
            "minLength" to jsonSchema { type(Kind.STRING); minLength(1) },
            "maxLength" to jsonSchema { type(Kind.STRING); maxLength(5) },
            "minItems=2" to jsonSchema { type(Kind.ARRAY); items { type(Kind.STRING) }; minItems(2) },
            "maxItems" to jsonSchema { type(Kind.ARRAY); items { type(Kind.STRING) }; maxItems(3) },
            "open object" to jsonSchema { type(Kind.OBJECT); additionalProperties { type(Kind.STRING) } },
            "lookahead" to jsonSchema { type(Kind.STRING); pattern("a(?=b)") },
            "backreference" to jsonSchema { type(Kind.STRING); pattern("(a)\\1") },
            "word boundary" to jsonSchema { type(Kind.STRING); pattern("\\bword") },
        )
        for ((name, bad) in cases) {
            val nested = jsonSchema { type(Kind.OBJECT); property("p", bad) }
            val outputError = assertFailsWith<InvalidRequestException>(name) { nested.toAnthropicOutputConfig(prompt) }
            assertTrue(outputError.message!!.contains("$.properties.p"), "$name: ${outputError.message}")
            assertFailsWith<InvalidRequestException>(name) { FunctionSchema("t", "d", nested).toAnthropicTool(prompt, strict = true) }
        }
    }

    @Test
    fun `constraint inside anyOf and array items is found with its path`() {
        val schema = jsonSchema {
            type(Kind.OBJECT)
            property("a", jsonSchema {
                type(Kind.ARRAY)
                items { anyOf(jsonSchema { type(Kind.STRING); maxLength(2) }, jsonSchema { type(Kind.BOOLEAN) }) }
            })
        }

        val error = assertFailsWith<InvalidRequestException> { schema.toAnthropicOutputConfig(prompt) }
        assertTrue(error.message!!.contains("$.properties.a.items.anyOf[0]: maxLength=2"), error.message)
    }

    @Serializable
    data class Verdict(
        @SchemaInfo("verdict") val ok: Boolean,
        @SchemaInfo("why") val reason: String? = null,
        @SchemaInfo("labels") val tags: List<String>,
    )

    @Test
    fun `typed settings become output_config format json_schema with closed objects`() {
        val config = wire(TypedPromptSettings(Verdict::class).toAnthropicOutputConfig(prompt))

        val format = config["format"]!!.jsonObject
        assertEquals("json_schema", format["type"].toString().trim('"'))
        val schema = format["schema"]!!.jsonObject
        assertEquals("false", schema["additionalProperties"].toString())
        assertEquals(setOf("ok", "reason", "tags"), schema["properties"]!!.jsonObject.keys)
        assertEquals("object", schema["type"].toString().trim('"'))
    }
}
