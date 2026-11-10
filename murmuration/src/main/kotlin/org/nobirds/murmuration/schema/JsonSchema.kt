package org.nobirds.murmuration.schema

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Provider-neutral sealed schema model. Roughly JSON Schema draft 7-ish, without any
 * provider-specific knobs. Each provider module supplies its own mapper from this model
 * to the SDK type (in `:murmuration-gemini` that's `JsonSchema.toGeminiSchema()`).
 *
 * Serialisation to a string for inclusion in a prompt goes through [toJson] on each
 * subtype — no kotlinx polymorphic discriminator, we want a clean JSON Schema object.
 */
sealed interface JsonSchema {
    val description: String?
    val nullable: Boolean

    /** JSON Schema form, for embedding in prompts and for debugging. */
    fun toJson(): JsonObject

    /** Convenience one-liner: a JSON Schema string. */
    fun toJsonString(): String = toJson().toString()
}

data class ObjectSchema(
    val properties: Map<String, JsonSchema> = emptyMap(),
    val required: List<String> = emptyList(),
    val additionalProperties: JsonSchema? = null,
    val title: String? = null,
    override val description: String? = null,
    override val nullable: Boolean = false,
) : JsonSchema {
    override fun toJson(): JsonObject = buildJsonObject {
        put("type", "object")
        if (properties.isNotEmpty()) {
            put("properties", buildJsonObject {
                properties.forEach { (k, v) -> put(k, v.toJson()) }
            })
        }
        if (required.isNotEmpty()) {
            put("required", buildJsonArray { required.forEach { add(it) } })
        }
        additionalProperties?.let { put("additionalProperties", it.toJson()) }
        title?.let { put("title", it) }
        description?.let { put("description", it) }
        if (nullable) put("nullable", true)
    }
}

data class ArraySchema(
    val items: JsonSchema,
    val minItems: Long? = null,
    val maxItems: Long? = null,
    override val description: String? = null,
    override val nullable: Boolean = false,
) : JsonSchema {
    override fun toJson(): JsonObject = buildJsonObject {
        put("type", "array")
        put("items", items.toJson())
        minItems?.let { put("minItems", it) }
        maxItems?.let { put("maxItems", it) }
        description?.let { put("description", it) }
        if (nullable) put("nullable", true)
    }
}

data class StringSchema(
    val format: String? = null,
    val pattern: String? = null,
    val minLength: Long? = null,
    val maxLength: Long? = null,
    val enumValues: List<String>? = null,
    val example: String? = null,
    override val description: String? = null,
    override val nullable: Boolean = false,
) : JsonSchema {
    override fun toJson(): JsonObject = buildJsonObject {
        put("type", "string")
        format?.let { put("format", it) }
        pattern?.let { put("pattern", it) }
        minLength?.let { put("minLength", it) }
        maxLength?.let { put("maxLength", it) }
        enumValues?.let { values ->
            put("enum", buildJsonArray { values.forEach { add(it) } })
        }
        example?.let { put("example", it) }
        description?.let { put("description", it) }
        if (nullable) put("nullable", true)
    }
}

data class IntegerSchema(
    /** "int32" / "int64" / null */
    val format: String? = null,
    val minValue: Long? = null,
    val maxValue: Long? = null,
    override val description: String? = null,
    override val nullable: Boolean = false,
) : JsonSchema {
    override fun toJson(): JsonObject = buildJsonObject {
        put("type", "integer")
        format?.let { put("format", it) }
        minValue?.let { put("minimum", it) }
        maxValue?.let { put("maximum", it) }
        description?.let { put("description", it) }
        if (nullable) put("nullable", true)
    }
}

data class NumberSchema(
    /** "float" / "double" / null */
    val format: String? = null,
    val minValue: Double? = null,
    val maxValue: Double? = null,
    override val description: String? = null,
    override val nullable: Boolean = false,
) : JsonSchema {
    override fun toJson(): JsonObject = buildJsonObject {
        put("type", "number")
        format?.let { put("format", it) }
        minValue?.let { put("minimum", it) }
        maxValue?.let { put("maximum", it) }
        description?.let { put("description", it) }
        if (nullable) put("nullable", true)
    }
}

data class BooleanSchema(
    override val description: String? = null,
    override val nullable: Boolean = false,
) : JsonSchema {
    override fun toJson(): JsonObject = buildJsonObject {
        put("type", "boolean")
        description?.let { put("description", it) }
        if (nullable) put("nullable", true)
    }
}

/**
 * `anyOf` — sealed polymorphism / polymorphic DTOs. The JSON Schema spec disallows other
 * constraints on top of `anyOf` (and Gemini/Vertex reject the combination), so we only
 * keep the list of variants and an optional description.
 */
data class AnyOfSchema(
    val variants: List<JsonSchema>,
    override val description: String? = null,
    override val nullable: Boolean = false,
) : JsonSchema {
    override fun toJson(): JsonObject = buildJsonObject {
        put("anyOf", buildJsonArray { variants.forEach { add(it.toJson()) } })
        description?.let { put("description", it) }
        if (nullable) put("nullable", true)
    }
}

/**
 * Standard `format` values for numeric types (plus `uuid` for strings). JSON Schema doesn't strictly normalise them,
 * but Gemini/Vertex and most other SDKs expect exactly these strings.
 */
object JsonSchemaFormats {
    const val INT32: String = "int32"
    const val INT64: String = "int64"
    const val FLOAT: String = "float"
    const val DOUBLE: String = "double"

    /** String `format` for canonical UUID text, used for `java.util.UUID` instrument parameters. */
    const val UUID: String = "uuid"
}
