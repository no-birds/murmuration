package org.nobirds.murmuration.gemini

import com.google.genai.types.Schema
import com.google.genai.types.Type
import org.nobirds.murmuration.schema.AnyOfSchema
import org.nobirds.murmuration.schema.ArraySchema
import org.nobirds.murmuration.schema.BooleanSchema
import org.nobirds.murmuration.schema.IntegerSchema
import org.nobirds.murmuration.schema.JsonSchema
import org.nobirds.murmuration.schema.NumberSchema
import org.nobirds.murmuration.schema.ObjectSchema
import org.nobirds.murmuration.schema.StringSchema

private val GEMINI_STRING_FORMATS = setOf("enum", "date-time")
private val GEMINI_INTEGER_FORMATS = setOf("int32", "int64")
private val GEMINI_NUMBER_FORMATS = setOf("float", "double")

/**
 * Splits a neutral `format` into what Gemini/Vertex may receive and what must be demoted to prose.
 * Gemini documents only a closed set of formats per type (STRING: `enum`, `date-time`; INTEGER:
 * `int32`, `int64`; NUMBER: `float`, `double`); anything else (e.g. `uuid`) is not sent in
 * `format` and becomes a hint appended to the description instead, so the model still sees it.
 */
private fun supportedFormatOrHint(format: String?, supported: Set<String>, description: String?): Pair<String?, String?> {
    if (format == null || format in supported) return format to description
    val hint = "format: $format"
    return null to (if (description.isNullOrEmpty()) hint else "$description ($hint)")
}

/**
 * Maps the provider-neutral [JsonSchema] to a Gemini SDK [Schema]. A plain recursive walk
 * over the sealed branches. No fields are added on top of [AnyOfSchema] — Gemini/Vertex
 * reject the combination (which is also why [AnyOfSchema] in the core carries only a short
 * set of fields). `format` is forwarded only when Gemini documents it for the type; other
 * formats are demoted to a description hint (see [supportedFormatOrHint]).
 */
fun JsonSchema.toGeminiSchema(): Schema {
    val builder = Schema.builder()
    when (this) {
        is ObjectSchema -> {
            builder.type(Type.Known.OBJECT)
            if (properties.isNotEmpty()) {
                builder.properties(properties.mapValues { it.value.toGeminiSchema() })
            }
            if (required.isNotEmpty()) {
                builder.required(required)
            }
            title?.let { builder.title(it) }
            description?.let { builder.description(it) }
            if (nullable) builder.nullable(true)
        }
        is ArraySchema -> {
            builder.type(Type.Known.ARRAY)
            builder.items(items.toGeminiSchema())
            minItems?.let { builder.minItems(it) }
            maxItems?.let { builder.maxItems(it) }
            description?.let { builder.description(it) }
            if (nullable) builder.nullable(true)
        }
        is StringSchema -> {
            builder.type(Type.Known.STRING)
            val (geminiFormat, geminiDescription) = supportedFormatOrHint(format, GEMINI_STRING_FORMATS, description)
            geminiFormat?.let { builder.format(it) }
            pattern?.let { builder.pattern(it) }
            minLength?.let { builder.minLength(it) }
            maxLength?.let { builder.maxLength(it) }
            enumValues?.let { builder.enum_(it) }
            example?.let { builder.example(it) }
            geminiDescription?.let { builder.description(it) }
            if (nullable) builder.nullable(true)
        }
        is IntegerSchema -> {
            builder.type(Type.Known.INTEGER)
            val (geminiFormat, geminiDescription) = supportedFormatOrHint(format, GEMINI_INTEGER_FORMATS, description)
            geminiFormat?.let { builder.format(it) }
            minValue?.let { builder.minimum(it.toDouble()) }
            maxValue?.let { builder.maximum(it.toDouble()) }
            geminiDescription?.let { builder.description(it) }
            if (nullable) builder.nullable(true)
        }
        is NumberSchema -> {
            builder.type(Type.Known.NUMBER)
            val (geminiFormat, geminiDescription) = supportedFormatOrHint(format, GEMINI_NUMBER_FORMATS, description)
            geminiFormat?.let { builder.format(it) }
            minValue?.let { builder.minimum(it) }
            maxValue?.let { builder.maximum(it) }
            geminiDescription?.let { builder.description(it) }
            if (nullable) builder.nullable(true)
        }
        is BooleanSchema -> {
            builder.type(Type.Known.BOOLEAN)
            description?.let { builder.description(it) }
            if (nullable) builder.nullable(true)
        }
        is AnyOfSchema -> {
            builder.anyOf(variants.map { it.toGeminiSchema() })
        }
    }
    return builder.build()
}
