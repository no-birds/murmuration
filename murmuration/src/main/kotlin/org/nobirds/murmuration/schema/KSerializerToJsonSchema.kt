@file:OptIn(ExperimentalSerializationApi::class)

package org.nobirds.murmuration.schema

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.KSerializer
import kotlinx.serialization.descriptors.PolymorphicKind
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.SerialKind
import kotlinx.serialization.descriptors.StructureKind
import kotlinx.serialization.json.JsonClassDiscriminator
import org.nobirds.murmuration.internal.UUIDSerializer
import org.nobirds.murmuration.tooling.NotBlank
import org.nobirds.murmuration.tooling.TransformerRegistry
import org.nobirds.murmuration.tooling.ValidatorRegistry
import kotlin.reflect.full.primaryConstructor

/**
 * Walks a kotlinx serializer's descriptor into a provider-neutral [JsonSchema]. Each
 * provider module (for example, `:murmuration-gemini`) maps the resulting [JsonSchema]
 * to its own SDK schema type.
 *
 * [extraAnnotations] are merged into the top-level descriptor's annotations so callers can
 * project parameter-level annotations (e.g. `@NotBlank` on a tool parameter) onto the
 * generated schema in addition to whatever lives on the type itself.
 */
fun <T> KSerializer<T>.toJsonSchema(
    extraAnnotations: List<Annotation> = emptyList(),
): JsonSchema =
    SerialElement(
        index = 0,
        name = descriptor.serialName,
        descriptor = descriptor,
        annotations = descriptor.annotations + extraAnnotations,
        isOptional = false,
    ).toJsonSchema()

val SerialDescriptor.indices: IntRange get() = 0 until elementsCount

fun SerialElement.elements(): Sequence<SerialElement> = sequence {
    for (i in descriptor.indices) {
        yield(
            SerialElement(
                i,
                descriptor.getElementName(i),
                descriptor.getElementDescriptor(i),
                descriptor.getElementAnnotations(i),
                descriptor.isElementOptional(i),
                this@elements,
            )
        )
    }
}

data class SerialElement(
    val index: Int,
    val name: String,
    val descriptor: SerialDescriptor,
    val annotations: List<Annotation>,
    val isOptional: Boolean,
    val parent: SerialElement? = null,
)

private fun JsonSchemaBuilder.applySchemaInfo(schemaInfo: SchemaInfo?) {
    if (schemaInfo == null) return
    if (schemaInfo.description.isNotBlank()) description(schemaInfo.description)
    if (schemaInfo.example.isNotBlank()) example(schemaInfo.example)
    if (schemaInfo.format.isNotBlank()) format(schemaInfo.format)
    if (schemaInfo.pattern.isNotBlank()) pattern(schemaInfo.pattern)
    if (schemaInfo.minLength != -1L) minLength(schemaInfo.minLength)
    if (schemaInfo.maxLength != -1L) maxLength(schemaInfo.maxLength)
    if (schemaInfo.minValue != Double.NEGATIVE_INFINITY) minimum(schemaInfo.minValue)
    if (schemaInfo.maxValue != Double.POSITIVE_INFINITY) maximum(schemaInfo.maxValue)
}

/**
 * Walks [annotations] through [ValidatorRegistry.Default] and [TransformerRegistry.Default]
 * (with [ValidationGroup] / [TransformationGroup] composing) and lets each resolved
 * validator project itself onto the schema. Validators' and transformers' textual
 * `describe()` summaries are merged into a single `description` line appended to whatever
 * description the schema already carries, so the LLM sees both the formal constraints and
 * a natural-language hint (e.g. "literal `\n` is normalized to a real newline" lets the
 * model emit either form safely).
 *
 * Transformers don't have an `applyToSchema` projection (they rewrite values, not constrain
 * them) — they contribute only to the description hints.
 */
private fun JsonSchemaBuilder.applyValidators(
    annotations: List<Annotation>,
    existingDescription: String?,
) {
    if (annotations.isEmpty()) return
    val validators = annotations.flatMap { ValidatorRegistry.Default.resolve(it) }
    val transformers = annotations.flatMap { TransformerRegistry.Default.resolve(it) }
    if (validators.isEmpty() && transformers.isEmpty()) return
    for (v in validators) v.applyToSchema(this)
    val hints = (validators.map { it.describe() } + transformers.map { it.describe() })
        .filter(String::isNotBlank)
    if (hints.isEmpty()) return
    val merged = if (existingDescription.isNullOrBlank()) hints.joinToString("; ")
        else "$existingDescription (${hints.joinToString("; ")})"
    description(merged)
}

private inline fun <reified T : Annotation> List<Annotation>.firstAnnotationOrNull(): T? =
    firstOrNull { it is T } as? T

fun List<Annotation>.findJsonSchemaInfo(): SchemaInfo? = firstAnnotationOrNull<SchemaInfo>()
fun List<Annotation>.findJsonGeneratorMeta(): SchemaGeneratorMeta? = firstAnnotationOrNull<SchemaGeneratorMeta>()

/**
 * A field is included in the parent object schema if it carries [SchemaInfo] OR any
 * annotation that resolves to at least one [ParameterValidator] or [ParameterTransformer]
 * (so a field tagged only with `@NotBlank` or only `@NormalizeNewlines` is still rendered
 * — the LLM still needs to see it in the schema).
 */
private fun SerialElement.shouldIncludeInSchema(): Boolean {
    if (annotations.findJsonSchemaInfo() != null) return true
    return annotations.any {
        ValidatorRegistry.Default.resolve(it).isNotEmpty() ||
            TransformerRegistry.Default.resolve(it).isNotEmpty()
    }
}

/**
 * Whether a property must appear in the parent object's `required` list — i.e. whether the
 * model is obliged to emit it.
 *
 * kotlinx's `isElementOptional` (there is a Kotlin default value) answers a DESERIALIZATION
 * question, not the LLM contract one. A tool-facing DTO routinely keeps a default (`= ""`)
 * so that a malformed call degrades into a friendly domain error instead of a raw kotlinx
 * `MissingFieldException` — while the field itself is mandatory for the caller. [NotBlank]
 * is precisely that declaration (its own `describe()` reads "required, must not be empty"),
 * so it is projected onto `required` as well as onto minLength/pattern. Without this a cheap
 * model happily omits the field: 2026-07-31 `gemini-3.1-flash-lite` shipped a whole batch of
 * seven `addAction` calls with no `textId` (optional in the schema, mandatory at the gate),
 * ate seven rejections and re-sent the identical batch — one wasted round-trip over an ~83K
 * token cache, systematically, twice per turn.
 *
 * Nullable properties are exempt: `null` is a legal value for them, so demanding presence
 * would contradict the type (and `@NotBlank` on a nullable field is a modelling error anyway).
 */
private fun SerialElement.isRequiredInSchema(): Boolean =
    !isOptional || (!descriptor.isNullable && annotations.any { it is NotBlank })

fun SerialElement.findSchemaGenerator(): SchemaGenerator? =
    annotations.findJsonGeneratorMeta()?.generator?.takeIf { it != Nothing::class }?.let {
        it.objectInstance ?: it.primaryConstructor?.call()
    }

data class SchemaDiscriminator(val discriminator: String, val name: String)

fun SerialElement.toJsonSchema(): JsonSchema {
    val generator = findSchemaGenerator()
    if (generator != null) return with(generator) { generate() }
    return generateDefault()
}

private fun SerialElement.generateDefault(): JsonSchema = jsonSchema {
    when (descriptor.kind) {
        PrimitiveKind.BOOLEAN -> {
            type(JsonSchemaBuilder.Kind.BOOLEAN)
        }
        PrimitiveKind.BYTE, PrimitiveKind.SHORT, PrimitiveKind.INT -> {
            type(JsonSchemaBuilder.Kind.INTEGER)
            format(JsonSchemaFormats.INT32)
        }
        PrimitiveKind.LONG -> {
            type(JsonSchemaBuilder.Kind.INTEGER)
            format(JsonSchemaFormats.INT64)
        }
        PrimitiveKind.FLOAT -> {
            type(JsonSchemaBuilder.Kind.NUMBER)
            format(JsonSchemaFormats.FLOAT)
        }
        PrimitiveKind.DOUBLE -> {
            type(JsonSchemaBuilder.Kind.NUMBER)
            format(JsonSchemaFormats.DOUBLE)
        }
        PrimitiveKind.CHAR, PrimitiveKind.STRING -> {
            type(JsonSchemaBuilder.Kind.STRING)
            // Nullable wrappers append "?" to the serial name.
            if (descriptor.serialName.removeSuffix("?") == UUIDSerializer.SERIAL_NAME) {
                format(JsonSchemaFormats.UUID)
            }
        }
        SerialKind.ENUM -> {
            type(JsonSchemaBuilder.Kind.STRING)
            enum(elements().map { it.name }.toList())
        }
        StructureKind.LIST -> {
            type(JsonSchemaBuilder.Kind.ARRAY)
            require(descriptor.elementsCount > 0) {
                "List must have at least one element to define item type. Element: ${this@generateDefault}"
            }
            items(elements().first().toJsonSchema())
        }
        StructureKind.CLASS, StructureKind.OBJECT -> {
            val discriminator = findDiscriminator()
            require(descriptor.elementsCount > 0 || discriminator != null) {
                "Class must have at least one property to define schema or discriminator must be specified. Element: ${this@generateDefault}"
            }

            type(JsonSchemaBuilder.Kind.OBJECT)

            if (discriminator != null) {
                property(discriminator.discriminator) {
                    type(JsonSchemaBuilder.Kind.STRING)
                    enum(discriminator.name)
                }
                required(discriminator.discriminator)
            }

            val included = elements().filter { it.shouldIncludeInSchema() }.toList()
            // Fail fast: a class whose fields are ALL filtered out by [shouldIncludeInSchema]
            // would ship an empty object schema (`properties={}`) to the provider. Under
            // constrained decoding the model can then emit nothing but `{}` — the tool call
            // deterministically fails (or silently loses data) at runtime. Better to blow up
            // at agent setup time with a pointer to the fix. (Bug 2026-06-06: un-annotated
            // `MemoryLink` made every `links=[...]` arrive as `[{}]`.)
            require(included.isNotEmpty() || discriminator != null) {
                "Class '${descriptor.serialName}' produces an EMPTY object schema: none of its " +
                    "${descriptor.elementsCount} properties carry @SchemaInfo or a validator/transformer " +
                    "annotation (see shouldIncludeInSchema). Annotate the fields the LLM must see — " +
                    "an empty object schema forces constrained generation to emit only {}."
            }
            for (element in included) {
                property(element.name, element.toJsonSchema())
                if (element.isRequiredInSchema()) {
                    required(element.name)
                }
            }
        }
        PolymorphicKind.SEALED -> {
            val directSubclasses = elements().filter {
                it.descriptor.kind == StructureKind.CLASS || it.descriptor.kind == StructureKind.OBJECT
            }.toList()

            val wrappedSubclasses = elements()
                .filter { it.descriptor.kind == SerialKind.CONTEXTUAL }
                .flatMap { it.elements() }
                .toList()

            val allSubclasses = directSubclasses + wrappedSubclasses
            val subSchemas = allSubclasses.map { it.toJsonSchema() }
            if (subSchemas.isNotEmpty()) {
                anyOf(subSchemas)
            }
            // Don't add any other fields on top of anyOf — Vertex/Gemini reject the combination.
            return@jsonSchema
        }
        else -> error("${descriptor.kind} is not supported as a root or property type. Element: ${this@generateDefault}")
    }

    nullable(descriptor.isNullable)
    val schemaInfo = annotations.findJsonSchemaInfo()
        ?: descriptor.annotations.findJsonSchemaInfo()
    applySchemaInfo(schemaInfo)
    // An enum's class-level [SchemaInfo] is the legend of its values (short serial names): it must reach the model
    // at every place, also where the place carries its own description, so it is appended instead of shadowed.
    val enumLegend = descriptor.takeIf { it.kind == SerialKind.ENUM }
        ?.annotations?.findJsonSchemaInfo()?.description?.takeIf { it.isNotBlank() }
    val ownDescription = schemaInfo?.description?.takeIf { it.isNotBlank() }
    val effectiveDescription = if (enumLegend != null && ownDescription != null && ownDescription != enumLegend) {
        "$ownDescription $enumLegend".also { description(it) }
    } else {
        ownDescription
    }
    applyValidators(annotations, effectiveDescription)
}

private fun SerialElement.findDiscriminator(): SchemaDiscriminator? {
    val p = parent ?: return null
    val pp = p.parent ?: return null

    if (pp.descriptor.kind == PolymorphicKind.SEALED && p.descriptor.kind == SerialKind.CONTEXTUAL) {
        val propertyName = pp.descriptor.sealedClassDiscriminator(p.annotations, pp.annotations)
        val propertyValue = name
        return SchemaDiscriminator(propertyName, propertyValue)
    }
    return null
}

/**
 * THE rule naming the discriminator property of a sealed hierarchy, shared by the schema
 * ([findDiscriminator]) and the tool-argument decoder (`ToolArgumentDecoder`), so the property the
 * model is told to send and the property the decoder reads cannot disagree. The first
 * [JsonClassDiscriminator] found in [annotationSources] (in order), then on this sealed descriptor
 * itself, wins; otherwise kotlinx's default `"type"`.
 */
internal fun SerialDescriptor.sealedClassDiscriminator(vararg annotationSources: List<Annotation>): String {
    val annotation = annotationSources.asSequence().plus(sequenceOf(annotations))
        .flatMap { it.asSequence() }
        .filterIsInstance<JsonClassDiscriminator>()
        .firstOrNull()
    return annotation?.discriminator ?: "type"
}

interface SchemaGenerator {
    fun SerialElement.generate(): JsonSchema

    companion object {
        val DEFAULT: SchemaGenerator = DefaultSchemaGenerator
    }
}

object DefaultSchemaGenerator : SchemaGenerator {
    override fun SerialElement.generate(): JsonSchema = toJsonSchema()
}

object PrimitiveUnionGenerator : SchemaGenerator {
    override fun SerialElement.generate(): JsonSchema = jsonSchema {
        anyOf(
            jsonSchema { type(JsonSchemaBuilder.Kind.STRING) },
            jsonSchema { type(JsonSchemaBuilder.Kind.INTEGER) },
            jsonSchema { type(JsonSchemaBuilder.Kind.NUMBER) },
            jsonSchema { type(JsonSchemaBuilder.Kind.BOOLEAN) },
        )
    }
}
