package org.nobirds.murmuration.openai

import com.openai.core.JsonValue
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.nobirds.murmuration.prompt.toAny
import org.nobirds.murmuration.schema.AnyOfSchema
import org.nobirds.murmuration.schema.ArraySchema
import org.nobirds.murmuration.schema.BooleanSchema
import org.nobirds.murmuration.schema.FunctionSchema
import org.nobirds.murmuration.schema.IntegerSchema
import org.nobirds.murmuration.schema.JsonSchema
import org.nobirds.murmuration.schema.NumberSchema
import org.nobirds.murmuration.schema.ObjectSchema
import org.nobirds.murmuration.schema.StringSchema

/**
 * Maps the provider-neutral [JsonSchema] to a JSON Schema `JsonObject` that Azure OpenAI
 * accepts for function parameters and structured outputs.
 *
 * This deliberately does **not** reuse [JsonSchema.toJson]: that emitter uses the OpenAPI
 * dialect (`nullable: true`), while OpenAI expects standard JSON Schema, where nullability
 * is a `type` union (`"type": ["string", "null"]`).
 *
 * [strict] selects the two OpenAI regimes:
 * - `strict = true` (structured output / `response_format.json_schema`): every object must
 *   set `"additionalProperties": false` and list **all** of its property keys in `required`.
 *   A property that was optional in the source schema (absent from its object's `required`)
 *   is instead expressed as nullable (its `type` gains `"null"`), so the model can return
 *   `null` in place of omitting the field — the only way to keep a field optional under
 *   OpenAI strict mode.
 * - `strict = false` (tool/function parameters): `required` is passed through as declared and
 *   `additionalProperties` is left unset (Azure does not require it for tool params; omitting
 *   keeps the emitted schema minimal).
 */
fun JsonSchema.toOpenAiJson(strict: Boolean): JsonObject = buildNode(strict, nullable)

/** JSON Schema `type` value: a bare string, or a `["<type>", "null"]` union when nullable. */
private fun typeValue(type: String, nullable: Boolean): JsonElement =
    if (nullable) JsonArray(listOf(JsonPrimitive(type), JsonPrimitive("null")))
    else JsonPrimitive(type)

private fun JsonSchema.buildNode(strict: Boolean, nullable: Boolean): JsonObject = buildJsonObject {
    when (val schema = this@buildNode) {
        is ObjectSchema -> {
            put("type", typeValue("object", nullable))
            if (schema.properties.isNotEmpty()) {
                put("properties", buildJsonObject {
                    schema.properties.forEach { (key, value) ->
                        // Under strict mode an optional property (not in the original
                        // `required`) becomes nullable so the model may return null; its own
                        // declared nullability is OR-ed in.
                        val childNullable = value.nullable || (strict && key !in schema.required)
                        put(key, value.buildNode(strict, childNullable))
                    }
                })
            }
            // Strict: required = ALL property keys; non-strict: required as declared.
            val requiredKeys = if (strict) schema.properties.keys.toList() else schema.required
            if (requiredKeys.isNotEmpty()) {
                put("required", buildJsonArray { requiredKeys.forEach { add(it) } })
            }
            if (strict) put("additionalProperties", false)
            schema.title?.let { put("title", it) }
            schema.description?.let { put("description", it) }
        }

        is ArraySchema -> {
            put("type", typeValue("array", nullable))
            put("items", schema.items.buildNode(strict, schema.items.nullable))
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
            // OpenAI has no `nullable` on `anyOf`; nullability becomes an extra `{"type":"null"}`
            // variant instead.
            put("anyOf", buildJsonArray {
                schema.variants.forEach { add(it.buildNode(strict, it.nullable)) }
                if (nullable) add(buildJsonObject { put("type", "null") })
            })
            schema.description?.let { put("description", it) }
        }
    }
}

/**
 * Maps a provider-neutral [FunctionSchema] to the JSON body Azure OpenAI expects for a tool:
 * `{ "name", "description", "parameters" }`. Tool parameters use the non-strict regime.
 */
fun FunctionSchema.toOpenAiFunctionJson(): JsonObject = buildJsonObject {
    put("name", name)
    put("description", description)
    put("parameters", parameters.toOpenAiJson(strict = false))
}

/**
 * Bridges a kotlinx [JsonObject] into the OpenAI SDK's untyped [JsonValue] map bag
 * ([com.openai.models.FunctionParameters] / [com.openai.models.ResponseFormatJsonSchema]'s
 * `Schema` are pure `Map<String, JsonValue>`). Conversion goes through [toAny], which turns
 * the JSON tree into plain Kotlin `Map`/`List`/primitives that [JsonValue.from] accepts.
 */
internal fun JsonObject.toJsonValue(): JsonValue = JsonValue.from(toAny())

/**
 * Splits a kotlinx [JsonObject] into the `Map<String, JsonValue>` bag that the OpenAI SDK's
 * `FunctionParameters` / `ResponseFormatJsonSchema.JsonSchema.Schema` builders expect via
 * `putAllAdditionalProperties`. Each top-level value is converted through [toAny] +
 * [JsonValue.from].
 */
internal fun JsonObject.toJsonValueMap(): Map<String, JsonValue> =
    mapValues { JsonValue.from(it.value.toAny()) }
