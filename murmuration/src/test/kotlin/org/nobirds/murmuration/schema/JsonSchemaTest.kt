package org.nobirds.murmuration.schema

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class JsonSchemaTest {

    private fun JsonObject.parse(): JsonObject = Json.parseToJsonElement(this.toString()).let { it as JsonObject }

    @Test
    fun `ObjectSchema toJson includes type properties required`() {
        val schema = ObjectSchema(
            properties = mapOf(
                "name" to StringSchema(),
                "age" to IntegerSchema(),
            ),
            required = listOf("name"),
            description = "user record",
        )
        val json = schema.toJson()
        assertEquals("object", (json["type"] as kotlinx.serialization.json.JsonPrimitive).content)
        assertTrue(json.containsKey("properties"))
        assertEquals("[\"name\"]", json["required"].toString())
        assertEquals("user record", (json["description"] as kotlinx.serialization.json.JsonPrimitive).content)
    }

    @Test
    fun `ObjectSchema with additionalProperties emits map-form`() {
        val schema = ObjectSchema(
            properties = emptyMap(),
            additionalProperties = StringSchema(),
        )
        val expected = buildJsonObject {
            put("type", "object")
            put("additionalProperties", buildJsonObject { put("type", "string") })
        }
        assertEquals(expected, schema.toJson())
    }

    @Test
    fun `ArraySchema serialises items and bounds`() {
        val schema = ArraySchema(items = StringSchema(), minItems = 1, maxItems = 10)
        val expected = buildJsonObject {
            put("type", "array")
            put("items", buildJsonObject { put("type", "string") })
            put("minItems", 1L)
            put("maxItems", 10L)
        }
        assertEquals(expected, schema.toJson())
    }

    @Test
    fun `StringSchema with enum and format`() {
        val schema = StringSchema(enumValues = listOf("A", "B"), format = "uuid")
        val json = schema.toJson()
        assertEquals("[\"A\",\"B\"]", json["enum"].toString())
        assertEquals("uuid", (json["format"] as kotlinx.serialization.json.JsonPrimitive).content)
    }

    @Test
    fun `IntegerSchema with format and bounds`() {
        val schema = IntegerSchema(format = JsonSchemaFormats.INT64, minValue = 0, maxValue = 100)
        val expected = buildJsonObject {
            put("type", "integer")
            put("format", "int64")
            put("minimum", 0L)
            put("maximum", 100L)
        }
        assertEquals(expected, schema.toJson())
    }

    @Test
    fun `NumberSchema with double format`() {
        val schema = NumberSchema(format = JsonSchemaFormats.DOUBLE, minValue = 0.0)
        val expected = buildJsonObject {
            put("type", "number")
            put("format", "double")
            put("minimum", 0.0)
        }
        assertEquals(expected, schema.toJson())
    }

    @Test
    fun `BooleanSchema nullable`() {
        val schema = BooleanSchema(nullable = true)
        val expected = buildJsonObject {
            put("type", "boolean")
            put("nullable", true)
        }
        assertEquals(expected, schema.toJson())
    }

    @Test
    fun `AnyOfSchema serialises variants`() {
        val schema = AnyOfSchema(variants = listOf(StringSchema(), IntegerSchema()))
        val json = schema.toJson()
        assertTrue(json.containsKey("anyOf"))
        assertTrue(!json.containsKey("type"), "AnyOfSchema must not emit a top-level type")
    }

    @Test
    fun `nullable flag emits when true and is omitted otherwise`() {
        val withNullable = StringSchema(nullable = true).toJson()
        assertEquals("true", withNullable["nullable"].toString())

        val withoutNullable = StringSchema(nullable = false).toJson()
        assertTrue(!withoutNullable.containsKey("nullable"))
    }
}
