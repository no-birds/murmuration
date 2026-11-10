@file:OptIn(ExperimentalSerializationApi::class)

package org.nobirds.murmuration.tooling

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialInfo
import org.nobirds.murmuration.schema.JsonSchemaBuilder

/**
 * Universal validation annotations and their bundled [ParameterValidator]s. Registered in
 * [ValidatorRegistry.Default]; no application-specific knowledge belongs here.
 *
 * Each annotation targets:
 * - [AnnotationTarget.VALUE_PARAMETER] — direct use on `@Instrument` function parameters.
 * - [AnnotationTarget.PROPERTY] — use on data-class fields. The annotation is also marked with
 *   kotlinx-serialization's [SerialInfo] so it is propagated through
 *   [kotlinx.serialization.descriptors.SerialDescriptor.getElementAnnotations] — without
 *   that, the schema walker would not see field-level validators on DTOs.
 * - [AnnotationTarget.ANNOTATION_CLASS] — so the annotation can be referenced from a
 *   composite shortcut declared with [ValidationGroup].
 *
 * ⚠️ ON A CONSTRUCTOR PROPERTY THE `@property:` USE-SITE IS MANDATORY:
 * ```
 * data class Draft(@property:NotBlank val textId: String = "")   // seen by the schema
 * data class Draft(@NotBlank val textId: String = "")            // SILENT NO-OP
 * ```
 * Because VALUE_PARAMETER is among the targets, Kotlin's default use-site for an annotation
 * written on a constructor property is `param` — the annotation lands on the constructor
 * parameter, never on the property, and therefore never reaches the serial descriptor. The
 * validator then contributes nothing: no minLength/pattern, no description hint, and (see
 * `isRequiredInSchema` in the schema walker) no `required` entry. That cost a systematic
 * double round-trip in production (2026-07-31, `addAction.textId`).
 */

/**
 * Marks an annotation as a "validation group" — a composing shortcut. The runtime walks
 * the annotations declared on the marked annotation class itself and collects validators
 * from each of them, with cycle protection. Use to build short, readable domain shortcuts:
 *
 * ```
 * @ValidationGroup
 * @NotBlank
 * @MaxLength(50)
 * @NotContains("(", ")")
 * annotation class CleanShortText
 * ```
 */
@Target(AnnotationTarget.ANNOTATION_CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class ValidationGroup

// --- String constraints ---

/** Forbids `null`, empty and whitespace-only values for `String` / `String?` parameters. */
@SerialInfo
@Target(AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.PROPERTY, AnnotationTarget.ANNOTATION_CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class NotBlank

/**
 * Forbids `null` and empty values. For `String` allows whitespace; for `Collection`,
 * `Map`, `Array` checks size ≥ 1. Use [NotBlank] when whitespace-only must also be
 * rejected for strings.
 */
@SerialInfo
@Target(AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.PROPERTY, AnnotationTarget.ANNOTATION_CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class NotEmpty

/** String length must be ≥ [value]. Skipped for `null`. */
@SerialInfo
@Target(AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.PROPERTY, AnnotationTarget.ANNOTATION_CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class MinLength(val value: Int)

/** String length must be ≤ [value]. Skipped for `null`. */
@SerialInfo
@Target(AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.PROPERTY, AnnotationTarget.ANNOTATION_CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class MaxLength(val value: Int)

/** String must match [regex] in full. Skipped for `null`. */
@SerialInfo
@Target(AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.PROPERTY, AnnotationTarget.ANNOTATION_CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class Pattern(val regex: String)

/** String must not contain any of the [forbidden] substrings. Skipped for `null`. */
@SerialInfo
@Target(AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.PROPERTY, AnnotationTarget.ANNOTATION_CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class NotContains(vararg val forbidden: String)

// --- Collection / Map / Array constraints ---

/** Collection / Map / Array size must be ≥ [value]. Skipped for `null`. */
@SerialInfo
@Target(AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.PROPERTY, AnnotationTarget.ANNOTATION_CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class MinSize(val value: Int)

/** Collection / Map / Array size must be ≤ [value]. Skipped for `null`. */
@SerialInfo
@Target(AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.PROPERTY, AnnotationTarget.ANNOTATION_CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class MaxSize(val value: Int)

// --- Numeric constraints ---

/** Numeric value must be ≥ [value]. Skipped for `null`. */
@SerialInfo
@Target(AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.PROPERTY, AnnotationTarget.ANNOTATION_CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class Min(val value: Long)

/** Numeric value must be ≤ [value]. Skipped for `null`. */
@SerialInfo
@Target(AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.PROPERTY, AnnotationTarget.ANNOTATION_CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class Max(val value: Long)

// --- Validators ---

internal object NotBlankValidator : ParameterValidator {
    override fun validate(name: String, value: Any?): ValidationResult {
        if (value == null) return reject(name, "must not be empty or blank")
        if (value !is String) return ValidationResult.Ok
        return if (value.isBlank()) reject(name, "must not be empty or blank")
        else ValidationResult.Ok
    }

    override fun applyToSchema(builder: JsonSchemaBuilder) {
        builder.minLength(1)
        // \S = at least one non-whitespace character anywhere; covers blank-only strings.
        builder.pattern("\\S")
    }

    override fun describe(): String = "required, must not be empty"
}

internal object NotEmptyValidator : ParameterValidator {
    override fun validate(name: String, value: Any?): ValidationResult = when (value) {
        null -> reject(name, "must not be null")
        is String -> if (value.isEmpty()) reject(name, "must not be an empty string") else ValidationResult.Ok
        is Collection<*> -> if (value.isEmpty()) reject(name, "must contain at least one element") else ValidationResult.Ok
        is Map<*, *> -> if (value.isEmpty()) reject(name, "must contain at least one entry") else ValidationResult.Ok
        is Array<*> -> if (value.isEmpty()) reject(name, "must contain at least one element") else ValidationResult.Ok
        else -> ValidationResult.Ok
    }

    override fun applyToSchema(builder: JsonSchemaBuilder) {
        // Builder picks the relevant field at build() time depending on the resolved Kind:
        // STRING uses minLength, ARRAY uses minItems. Setting both is safe.
        builder.minLength(1)
        builder.minItems(1)
    }

    override fun describe(): String = "must not be empty"
}

internal class MinLengthValidator(private val min: Int) : ParameterValidator {
    override fun validate(name: String, value: Any?): ValidationResult {
        val s = value as? String ?: return ValidationResult.Ok
        return if (s.length < min) reject(name, "must be at least $min characters long (got ${s.length})")
        else ValidationResult.Ok
    }

    override fun applyToSchema(builder: JsonSchemaBuilder) {
        builder.minLength(min.toLong())
    }

    override fun describe(): String = "at least $min characters"
}

internal class MaxLengthValidator(private val max: Int) : ParameterValidator {
    override fun validate(name: String, value: Any?): ValidationResult {
        val s = value as? String ?: return ValidationResult.Ok
        return if (s.length > max) reject(name, "must be at most $max characters long (got ${s.length})")
        else ValidationResult.Ok
    }

    override fun applyToSchema(builder: JsonSchemaBuilder) {
        builder.maxLength(max.toLong())
    }

    override fun describe(): String = "at most $max characters"
}

internal class PatternValidator(private val pattern: String) : ParameterValidator {
    private val compiled = Regex(pattern)
    override fun validate(name: String, value: Any?): ValidationResult {
        val s = value as? String ?: return ValidationResult.Ok
        return if (!compiled.matches(s)) reject(name, "must match pattern `$pattern`")
        else ValidationResult.Ok
    }

    override fun applyToSchema(builder: JsonSchemaBuilder) {
        builder.pattern(pattern)
    }

    override fun describe(): String = "matches pattern `$pattern`"
}

internal class NotContainsValidator(private val forbidden: List<String>) : ParameterValidator {
    override fun validate(name: String, value: Any?): ValidationResult {
        val s = value as? String ?: return ValidationResult.Ok
        val hit = forbidden.firstOrNull { it.isNotEmpty() && s.contains(it) } ?: return ValidationResult.Ok
        return reject(name, "must not contain `$hit`")
    }

    // No applyToSchema: JSON-Schema has no native "not contains" operator and provider
    // regex support for negative-lookahead is uneven. The constraint stays a runtime
    // safety net; the LLM learns the rule via [describe] mixed into the field description.

    override fun describe(): String = forbidden
        .filter { it.isNotEmpty() }
        .joinToString(", ", prefix = "must not contain ") { "`$it`" }
}

internal class MinSizeValidator(private val min: Int) : ParameterValidator {
    override fun validate(name: String, value: Any?): ValidationResult {
        val size = sizeOrNull(value) ?: return ValidationResult.Ok
        return if (size < min) reject(name, "must contain at least $min items (got $size)")
        else ValidationResult.Ok
    }

    override fun applyToSchema(builder: JsonSchemaBuilder) {
        builder.minItems(min.toLong())
    }

    override fun describe(): String = "at least $min items"
}

internal class MaxSizeValidator(private val max: Int) : ParameterValidator {
    override fun validate(name: String, value: Any?): ValidationResult {
        val size = sizeOrNull(value) ?: return ValidationResult.Ok
        return if (size > max) reject(name, "must contain at most $max items (got $size)")
        else ValidationResult.Ok
    }

    override fun applyToSchema(builder: JsonSchemaBuilder) {
        builder.maxItems(max.toLong())
    }

    override fun describe(): String = "at most $max items"
}

internal class MinValidator(private val min: Long) : ParameterValidator {
    override fun validate(name: String, value: Any?): ValidationResult {
        val n = numberOrNull(value) ?: return ValidationResult.Ok
        return if (n < min) reject(name, "must be ≥ $min (got $n)")
        else ValidationResult.Ok
    }

    override fun applyToSchema(builder: JsonSchemaBuilder) {
        builder.minimum(min)
    }

    override fun describe(): String = "minimum $min"
}

internal class MaxValidator(private val max: Long) : ParameterValidator {
    override fun validate(name: String, value: Any?): ValidationResult {
        val n = numberOrNull(value) ?: return ValidationResult.Ok
        return if (n > max) reject(name, "must be ≤ $max (got $n)")
        else ValidationResult.Ok
    }

    override fun applyToSchema(builder: JsonSchemaBuilder) {
        builder.maximum(max)
    }

    override fun describe(): String = "maximum $max"
}

private fun reject(name: String, what: String): ValidationResult.Reject =
    ValidationResult.Reject("ERROR: parameter '$name' $what.")

private fun sizeOrNull(value: Any?): Int? = when (value) {
    is Collection<*> -> value.size
    is Map<*, *> -> value.size
    is Array<*> -> value.size
    is String -> value.length
    else -> null
}

private fun numberOrNull(value: Any?): Long? = when (value) {
    is Number -> value.toLong()
    else -> null
}
