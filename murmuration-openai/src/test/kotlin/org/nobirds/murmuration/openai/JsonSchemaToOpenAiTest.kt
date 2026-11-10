package org.nobirds.murmuration.openai

import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.schema.AnyOfSchema
import org.nobirds.murmuration.schema.ArraySchema
import org.nobirds.murmuration.schema.BooleanSchema
import org.nobirds.murmuration.schema.FunctionSchema
import org.nobirds.murmuration.schema.IntegerSchema
import org.nobirds.murmuration.schema.JsonSchemaFormats
import org.nobirds.murmuration.schema.NumberSchema
import org.nobirds.murmuration.schema.ObjectSchema
import org.nobirds.murmuration.schema.StringSchema
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 * Unit tests for [JsonSchema.toOpenAiJson] — the provider-neutral -> OpenAI JSON Schema mapper.
 * These build [org.nobirds.murmuration.schema.JsonSchema] trees directly (bypassing the
 * reflection-based [org.nobirds.murmuration.schema.toFunctionSchema] machinery) so they assert
 * OUR mapping logic in isolation, not the serializer/annotation pipeline that feeds it.
 */
class JsonSchemaToOpenAiTest {

    @Test
    fun `primitive schemas carry type, format and constraints`() {
        val string = StringSchema(format = "date-time", minLength = 1, maxLength = 10, pattern = "^a")
        val stringJson = string.toOpenAiJson(strict = false)
        assertEquals(JsonPrimitive("string"), stringJson["type"])
        assertEquals("date-time", stringJson["format"]?.jsonPrimitive?.content)
        assertEquals(1L, stringJson["minLength"]?.jsonPrimitive?.content?.toLong())
        assertEquals(10L, stringJson["maxLength"]?.jsonPrimitive?.content?.toLong())
        assertEquals("^a", stringJson["pattern"]?.jsonPrimitive?.content)

        val integer = IntegerSchema(format = JsonSchemaFormats.INT32, minValue = 0, maxValue = 100)
        val integerJson = integer.toOpenAiJson(strict = false)
        assertEquals(JsonPrimitive("integer"), integerJson["type"])
        assertEquals(JsonSchemaFormats.INT32, integerJson["format"]?.jsonPrimitive?.content)
        assertEquals(0L, integerJson["minimum"]?.jsonPrimitive?.content?.toLong())
        assertEquals(100L, integerJson["maximum"]?.jsonPrimitive?.content?.toLong())

        val number = NumberSchema(format = JsonSchemaFormats.DOUBLE, minValue = 0.5, maxValue = 9.5)
        val numberJson = number.toOpenAiJson(strict = false)
        assertEquals(JsonPrimitive("number"), numberJson["type"])
        assertEquals(0.5, numberJson["minimum"]?.jsonPrimitive?.content?.toDouble())
        assertEquals(9.5, numberJson["maximum"]?.jsonPrimitive?.content?.toDouble())

        val boolean = BooleanSchema()
        assertEquals(JsonPrimitive("boolean"), boolean.toOpenAiJson(strict = false)["type"])
    }

    @Test
    fun `string enum values are passed through`() {
        val schema = StringSchema(enumValues = listOf("a", "b", "c"))
        val json = schema.toOpenAiJson(strict = false)
        val enumArray = json["enum"]?.jsonArray
        assertEquals(listOf("a", "b", "c"), enumArray?.map { it.jsonPrimitive.content })
    }

    @Test
    fun `nullable primitive becomes a type union with null`() {
        val schema = StringSchema(nullable = true)
        val json = schema.toOpenAiJson(strict = false)
        val type = json["type"]?.jsonArray
        assertEquals(listOf("string", "null"), type?.map { it.jsonPrimitive.content })
    }

    @Test
    fun `nullable AnyOf adds a null variant instead of a type union`() {
        val schema = AnyOfSchema(variants = listOf(StringSchema(), IntegerSchema()), nullable = true)
        val json = schema.toOpenAiJson(strict = false)
        val variants = json["anyOf"]!!.jsonArray
        assertEquals(3, variants.size)
        assertEquals(JsonPrimitive("null"), variants.last().jsonObject["type"])
        // Nullability on anyOf is expressed ONLY as the extra variant — no top-level "nullable"
        // or "type" key (OpenAI has no `nullable` keyword and `anyOf` carries no sibling `type`).
        assertNull(json["nullable"])
        assertNull(json["type"])
    }

    @Test
    fun `audience union keeps shortcut enum and non-empty UUID array`() {
        val schema = AnyOfSchema(
            variants = listOf(
                StringSchema(enumValues = listOf("all", "nobody", "party")),
                ArraySchema(items = StringSchema(format = "uuid"), minItems = 1),
            ),
        )
        val json = schema.toOpenAiJson(strict = false)
        val variants = json["anyOf"]!!.jsonArray

        assertNull(json["type"])
        assertEquals(
            listOf("all", "nobody", "party"),
            variants[0].jsonObject["enum"]!!.jsonArray.map { it.jsonPrimitive.content },
        )
        assertEquals("array", variants[1].jsonObject["type"]!!.jsonPrimitive.content)
        assertEquals(1L, variants[1].jsonObject["minItems"]!!.jsonPrimitive.content.toLong())
        val item = variants[1].jsonObject["items"]!!.jsonObject
        assertEquals("string", item["type"]!!.jsonPrimitive.content)
        assertEquals("uuid", item["format"]!!.jsonPrimitive.content)
    }

    @Test
    fun `non-strict object keeps required as declared and omits additionalProperties`() {
        val schema = ObjectSchema(
            properties = mapOf("a" to StringSchema(), "b" to IntegerSchema()),
            required = listOf("a"),
        )
        val json = schema.toOpenAiJson(strict = false)
        assertEquals(listOf("a"), json["required"]?.jsonArray?.map { it.jsonPrimitive.content })
        assertNull(json["additionalProperties"])
        // "b" was optional and non-strict — must stay a plain (non-nullable) type.
        val bType = json["properties"]!!.jsonObject["b"]!!.jsonObject["type"]
        assertEquals(JsonPrimitive("integer"), bType)
    }

    @Test
    fun `strict object marks all properties required and makes optional ones nullable`() {
        val schema = ObjectSchema(
            properties = mapOf("a" to StringSchema(), "b" to IntegerSchema()),
            required = listOf("a"),
        )
        val json = schema.toOpenAiJson(strict = true)
        assertEquals(setOf("a", "b"), json["required"]?.jsonArray?.map { it.jsonPrimitive.content }?.toSet())
        assertEquals(JsonPrimitive(false), json["additionalProperties"])

        // "a" was already required — stays a plain string.
        assertEquals(JsonPrimitive("string"), json["properties"]!!.jsonObject["a"]!!.jsonObject["type"])
        // "b" was optional — becomes nullable (type union) under strict mode.
        val bType = json["properties"]!!.jsonObject["b"]!!.jsonObject["type"]!!.jsonArray
        assertEquals(listOf("integer", "null"), bType.map { it.jsonPrimitive.content })
    }

    @Test
    fun `strict mode propagates additionalProperties false to nested objects`() {
        val nested = ObjectSchema(properties = mapOf("x" to StringSchema()), required = listOf("x"))
        val schema = ObjectSchema(properties = mapOf("nested" to nested), required = listOf("nested"))
        val json = schema.toOpenAiJson(strict = true)
        val nestedJson = json["properties"]!!.jsonObject["nested"]!!.jsonObject
        assertEquals(JsonPrimitive(false), nestedJson["additionalProperties"])
        assertEquals(listOf("x"), nestedJson["required"]?.jsonArray?.map { it.jsonPrimitive.content })
    }

    @Test
    fun `function schema maps name, description and non-strict parameters`() {
        val function = FunctionSchema(
            name = "get_weather",
            description = "Look up the weather",
            parameters = ObjectSchema(
                properties = mapOf("city" to StringSchema(), "days" to IntegerSchema()),
                required = listOf("city"),
            ),
        )
        val json = function.toOpenAiFunctionJson()
        assertEquals("get_weather", json["name"]?.jsonPrimitive?.content)
        assertEquals("Look up the weather", json["description"]?.jsonPrimitive?.content)
        val parameters = json["parameters"]!!.jsonObject
        assertEquals(listOf("city"), parameters["required"]?.jsonArray?.map { it.jsonPrimitive.content })
        // Tool parameters use the non-strict regime — no additionalProperties key forced.
        assertNull(parameters["additionalProperties"])
    }
}
