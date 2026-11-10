package org.nobirds.murmuration.schema

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Emitter of **standard JSON Schema** from the provider-neutral [JsonSchema].
 *
 * It deliberately differs from [JsonSchema.toJson], which uses the OpenAPI dialect
 * (`nullable: true`) that Gemini/Vertex expect. Here nullability is a `type` union
 * (`"type": ["string", "null"]`), and a nullable `anyOf` gets an extra `{"type": "null"}` variant.
 * Providers that speak standard JSON Schema (OpenAI, Anthropic) build their dialect on top of this.
 *
 * Fidelity notes: [ObjectSchema.additionalProperties] is not emitted (only [closedObjects]
 * controls `additionalProperties`), and `required` is passed through exactly as declared.
 *
 * @param closedObjects when true, every object node gets `"additionalProperties": false`.
 */
fun JsonSchema.toStandardJson(closedObjects: Boolean = false): JsonObject = buildNode(closedObjects, nullable)

/** JSON Schema `type` value: a bare string, or a `["<type>", "null"]` union when nullable. */
private fun typeValue(type: String, nullable: Boolean): JsonElement =
    if (nullable) JsonArray(listOf(JsonPrimitive(type), JsonPrimitive("null")))
    else JsonPrimitive(type)

private fun JsonSchema.buildNode(closedObjects: Boolean, nullable: Boolean): JsonObject = buildJsonObject {
    when (val schema = this@buildNode) {
        is ObjectSchema -> {
            put("type", typeValue("object", nullable))
            if (schema.properties.isNotEmpty()) {
                put("properties", buildJsonObject {
                    schema.properties.forEach { (key, value) ->
                        put(key, value.buildNode(closedObjects, value.nullable))
                    }
                })
            }
            if (schema.required.isNotEmpty()) {
                put("required", buildJsonArray { schema.required.forEach { add(it) } })
            }
            if (closedObjects) put("additionalProperties", false)
            schema.title?.let { put("title", it) }
            schema.description?.let { put("description", it) }
        }

        is ArraySchema -> {
            put("type", typeValue("array", nullable))
            put("items", schema.items.buildNode(closedObjects, schema.items.nullable))
            schema.minItems?.let { put("minItems", it) }
            schema.maxItems?.let { put("maxItems", it) }
            schema.description?.let { put("description", it) }
        }

        is StringSchema -> {
            put("type", typeValue("string", nullable))
            schema.format?.let { put("format", it) }
            schema.pattern?.let { put("pattern", it) }
            schema.minLength?.let { put("minLength", it) }
            schema.maxLength?.let { put("maxLength", it) }
            schema.enumValues?.let { values ->
                put("enum", buildJsonArray { values.forEach { add(it) } })
            }
            schema.description?.let { put("description", it) }
        }

        is IntegerSchema -> {
            put("type", typeValue("integer", nullable))
            schema.format?.let { put("format", it) }
            schema.minValue?.let { put("minimum", it) }
            schema.maxValue?.let { put("maximum", it) }
            schema.description?.let { put("description", it) }
        }

        is NumberSchema -> {
            put("type", typeValue("number", nullable))
            schema.format?.let { put("format", it) }
            schema.minValue?.let { put("minimum", it) }
            schema.maxValue?.let { put("maximum", it) }
            schema.description?.let { put("description", it) }
        }

        is BooleanSchema -> {
            put("type", typeValue("boolean", nullable))
            schema.description?.let { put("description", it) }
        }

        is AnyOfSchema -> {
            // Standard JSON Schema has no `nullable` on `anyOf`; nullability becomes an extra
            // `{"type":"null"}` variant instead.
            put("anyOf", buildJsonArray {
                schema.variants.forEach { add(it.buildNode(closedObjects, it.nullable)) }
                if (nullable) add(buildJsonObject { put("type", "null") })
            })
            schema.description?.let { put("description", it) }
        }
    }
}
