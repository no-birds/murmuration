package org.nobirds.murmuration.anthropic

import org.nobirds.murmuration.schema.AnyOfSchema
import org.nobirds.murmuration.schema.ArraySchema
import org.nobirds.murmuration.schema.BooleanSchema
import org.nobirds.murmuration.schema.IntegerSchema
import org.nobirds.murmuration.schema.JsonSchema
import org.nobirds.murmuration.schema.NumberSchema
import org.nobirds.murmuration.schema.ObjectSchema
import org.nobirds.murmuration.schema.StringSchema

/**
 * The subset of JSON Schema that Claude accepts for strict schemas (structured outputs and `strict` tools).
 * Anything outside it makes the API answer 400, so it is detected here, before the network.
 *
 * Constructs the API rejects: `minimum`/`maximum`, `minLength`/`maxLength`, `minItems` other than 0 or 1, `maxItems`,
 * an open object (`additionalProperties` other than `false`), lookahead/lookbehind/backreference/word boundary in
 * `pattern`.
 *
 * Recursion and external `$ref` are rejected by the API too, but [JsonSchema] is a finite tree without references and
 * the standard emitter never writes `$ref`, so those states cannot be expressed here.
 */
internal object AnthropicSchemaSubset {

    private val unsupportedRegex = Regex("""\(\?[=!]|\(\?<[=!]|\\[1-9]|\\[bB]""")

    /** Human-readable violations of the subset in [schema], each prefixed with the JSON path; empty when it fits. */
    fun violations(schema: JsonSchema): List<String> = buildList { collect(schema, "$", this) }

    private fun collect(schema: JsonSchema, path: String, out: MutableList<String>) {
        when (schema) {
            is ObjectSchema -> {
                if (schema.additionalProperties != null) out += "$path: open object (additionalProperties must be false)"
                schema.properties.forEach { (name, child) -> collect(child, "$path.properties.$name", out) }
            }

            is ArraySchema -> {
                schema.minItems?.takeIf { it > 1 }?.let { out += "$path: minItems=$it (only 0 or 1 is supported)" }
                schema.maxItems?.let { out += "$path: maxItems=$it is not supported" }
                collect(schema.items, "$path.items", out)
            }

            is StringSchema -> {
                schema.minLength?.let { out += "$path: minLength=$it is not supported" }
                schema.maxLength?.let { out += "$path: maxLength=$it is not supported" }
                schema.pattern?.takeIf { unsupportedRegex.containsMatchIn(it) }
                    ?.let { out += "$path: pattern '$it' uses lookaround, backreference or word boundary" }
            }

            is IntegerSchema -> {
                schema.minValue?.let { out += "$path: minimum=$it is not supported" }
                schema.maxValue?.let { out += "$path: maximum=$it is not supported" }
            }

            is NumberSchema -> {
                schema.minValue?.let { out += "$path: minimum=$it is not supported" }
                schema.maxValue?.let { out += "$path: maximum=$it is not supported" }
            }

            is BooleanSchema -> Unit

            is AnyOfSchema -> schema.variants.forEachIndexed { i, v -> collect(v, "$path.anyOf[$i]", out) }
        }
    }
}
