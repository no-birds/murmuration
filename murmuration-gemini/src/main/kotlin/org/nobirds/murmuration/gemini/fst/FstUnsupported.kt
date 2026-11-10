package org.nobirds.murmuration.gemini.fst

import com.google.genai.types.FunctionDeclaration
import com.google.genai.types.Schema
import com.google.genai.types.Type

/**
 * Schema constraints the FST model has no measured treatment for. A place using one must either
 * be added to the model together with a Vertex measurement, or not be used.
 *
 * Deliberately NOT listed: `minLength`, `maxLength`, `pattern`, numeric `format`
 * (`int32`/`int64`/`float`/`double`). They occur in the measured profiles and the model, which
 * ignores them, matches Vertex on those profiles.
 *
 * Integer ranges are NOT free: the model reads only `minimum >= 0` / `>= 1` and ignores `maximum`,
 * while Vertex charges for wide ranges (kb agent-framework.md, price table: `0..7` +1, `0..100` +3,
 * `+-1e6` +10). Ranges measured against Vertex (`0..9`, `1..20`) are within the +-1 tolerance, so
 * the model is trusted only on them: `minimum` must be 0 or 1 and `maximum` at most [MAX_MEASURED_MAXIMUM].
 * A wider range must be measured on Vertex and added to the model first.
 *
 * `minItems` is read by the model only as `< 1` / `>= 1`; `minItems >= 2` is refused, otherwise
 * the model undercounts the states of a required prefix of several items.
 */
internal object FstUnsupported {
    private val NUMERIC_FORMATS = setOf("int32", "int64", "float", "double")

    /** Largest integer `maximum` present in the measured profiles (`1..20`). */
    private const val MAX_MEASURED_MAXIMUM = 20.0
    private val MEASURED_MINIMUMS = setOf(0.0, 1.0)

    fun check(declaration: FunctionDeclaration) {
        val name = declaration.name().orElse("?")
        declaration.parameters().ifPresent { check(it, "$name.parameters") }
    }

    private fun check(schema: Schema, place: String) {
        val type = schema.type().orElse(null)?.knownEnum() ?: Type.Known.STRING
        require(schema.maxItems().isEmpty) { "FST model: maxItems is not modelled at $place" }
        schema.minItems().ifPresent {
            require(it <= 1) {
                "FST model: minItems $it at $place is not modelled (only 0 or 1); " +
                    "measure it on Vertex and add it to the model"
            }
        }
        schema.format().ifPresent {
            require(it in NUMERIC_FORMATS) { "FST model: format '$it' is not modelled at $place" }
        }
        require(schema.minimum().isEmpty || type == Type.Known.INTEGER) {
            "FST model: minimum is modelled only for INTEGER, found at $place"
        }
        require(schema.maximum().isEmpty || type == Type.Known.INTEGER) {
            "FST model: maximum is modelled only for INTEGER, found at $place"
        }
        schema.minimum().ifPresent {
            require(it in MEASURED_MINIMUMS) {
                "FST model: minimum $it at $place is outside the measured set $MEASURED_MINIMUMS; " +
                    "measure it on Vertex and add it to the model"
            }
        }
        schema.maximum().ifPresent {
            require(it <= MAX_MEASURED_MAXIMUM) {
                "FST model: maximum $it at $place exceeds the measured $MAX_MEASURED_MAXIMUM; " +
                    "wide integer ranges cost states the model does not count (kb: 0..100 +3, 1e6 +10)"
            }
        }
        require(schema.enum_().map { it.isEmpty() }.orElse(true) || type == Type.Known.STRING) {
            "FST model: enum is modelled only for STRING, found at $place"
        }
        schema.properties().ifPresent { props -> props.forEach { (k, v) -> check(v, "$place.$k") } }
        schema.items().ifPresent { check(it, "$place[]") }
        schema.anyOf().ifPresent { list -> list.forEachIndexed { i, v -> check(v, "$place.anyOf[$i]") } }
    }
}
