package org.nobirds.murmuration.schema

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialInfo
import kotlin.reflect.KClass

/**
 * Per-property (or per-class) metadata for schema generation. The walker
 * ([toJsonSchema][org.nobirds.murmuration.schema.toJsonSchema]) reads these and forwards
 * them to the matching [JsonSchema] subtype. Defaults mean "not set" and are skipped.
 */
@OptIn(ExperimentalSerializationApi::class)
@SerialInfo
@Retention(AnnotationRetention.RUNTIME)
@Target(AnnotationTarget.PROPERTY, AnnotationTarget.CLASS) // also allowed on a class to describe the whole schema
annotation class SchemaInfo(
    val description: String = "",
    val format: String = "",                                   // explicit format hint (e.g. "uuid", "date-time")
    val example: String = "",                                  // not used in schema directly, but useful for docs
    val minLength: Long = -1,                                  // string min length
    val maxLength: Long = -1,                                  // string max length
    val minValue: Double = Double.NEGATIVE_INFINITY,           // numeric minimum
    val maxValue: Double = Double.POSITIVE_INFINITY,           // numeric maximum
    val pattern: String = "",                                  // regex for string validation
)

/**
 * Routes a property through a custom [SchemaGenerator] implementation. The walker
 * instantiates the generator (object instance or no-arg constructor) and lets it produce
 * the [JsonSchema] for that property. Use this when the default kotlinx-driven generation
 * doesn't fit — e.g. union types like [PrimitiveUnionGenerator].
 *
 * On a primary-constructor property, use the explicit `@property:SchemaGeneratorMeta` use-site.
 * Without it Kotlin selects the value-parameter target, which is correct for instrument
 * parameters but is not included in the kotlinx serialization property descriptor.
 */
@OptIn(ExperimentalSerializationApi::class)
@SerialInfo
@Retention(AnnotationRetention.RUNTIME)
@Target(AnnotationTarget.PROPERTY, AnnotationTarget.VALUE_PARAMETER)
annotation class SchemaGeneratorMeta(
    val generator: KClass<out SchemaGenerator> = Nothing::class,
)
