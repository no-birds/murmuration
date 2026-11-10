package org.nobirds.murmuration.schema

/**
 * DSL builder for [JsonSchema]. Reads like a typical `schema { ... }` builder but produces
 * the provider-neutral model — [toJsonSchema] then maps it to whatever a provider expects.
 */
fun jsonSchema(block: JsonSchemaBuilder.() -> Unit): JsonSchema {
    val builder = JsonSchemaBuilder()
    builder.block()
    return builder.build()
}

class JsonSchemaBuilder {

    enum class Kind { OBJECT, ARRAY, STRING, INTEGER, NUMBER, BOOLEAN }

    private var kind: Kind? = null
    private var description: String? = null
    private var nullable: Boolean = false

    private val properties = mutableMapOf<String, JsonSchema>()
    private val required = mutableListOf<String>()
    private var additionalProperties: JsonSchema? = null
    private var title: String? = null

    private var items: JsonSchema? = null
    private var minItems: Long? = null
    private var maxItems: Long? = null

    private var format: String? = null
    private var pattern: String? = null
    private var minLength: Long? = null
    private var maxLength: Long? = null
    private var enumValues: MutableList<String>? = null
    private var example: String? = null

    // Numeric bounds are stored as Double (the largest common type). At build time for an
    // IntegerSchema we convert back to Long. Lossless enough in practice (above 2^53
    // Double loses precision, but bounds that large in a schema are exceedingly rare).
    private var minValue: Double? = null
    private var maxValue: Double? = null

    private val anyOfVariants = mutableListOf<JsonSchema>()

    fun type(t: Kind) { kind = t }
    fun description(s: String) { description = s }
    fun title(s: String) { title = s }
    fun nullable(b: Boolean) { nullable = b }

    fun format(s: String) { format = s }
    fun pattern(s: String) { pattern = s }
    fun example(s: String) { example = s }

    fun minLength(v: Long) { minLength = v }
    fun maxLength(v: Long) { maxLength = v }

    fun minimum(v: Long) { minValue = v.toDouble() }
    fun minimum(v: Double) { minValue = v }
    fun maximum(v: Long) { maxValue = v.toDouble() }
    fun maximum(v: Double) { maxValue = v }

    fun enum(vararg values: String) { enum(values.toList()) }
    fun enum(values: List<String>) {
        enumValues = (enumValues ?: mutableListOf()).also { it.addAll(values) }
    }

    fun items(s: JsonSchema) { items = s }
    fun items(block: JsonSchemaBuilder.() -> Unit) { items = jsonSchema(block) }
    fun minItems(v: Long) { minItems = v }
    fun maxItems(v: Long) { maxItems = v }

    fun property(name: String, schema: JsonSchema) { properties[name] = schema }
    fun property(name: String, block: JsonSchemaBuilder.() -> Unit) {
        properties[name] = jsonSchema(block)
    }
    fun required(vararg names: String) { required(names.toList()) }
    fun required(names: List<String>) { required += names }
    fun additionalProperties(schema: JsonSchema) { additionalProperties = schema }
    fun additionalProperties(block: JsonSchemaBuilder.() -> Unit) {
        additionalProperties = jsonSchema(block)
    }

    fun anyOf(vararg variants: JsonSchema) { anyOfVariants += variants }
    fun anyOf(variants: List<JsonSchema>) { anyOfVariants += variants }

    fun build(): JsonSchema {
        if (anyOfVariants.isNotEmpty()) {
            return AnyOfSchema(
                variants = anyOfVariants.toList(),
                description = description,
                nullable = nullable,
            )
        }
        return when (kind) {
            Kind.OBJECT -> ObjectSchema(
                properties = properties.toMap(),
                required = required.toList(),
                additionalProperties = additionalProperties,
                title = title,
                description = description,
                nullable = nullable,
            )
            Kind.ARRAY -> ArraySchema(
                items = items ?: error("ArraySchema needs items"),
                minItems = minItems,
                maxItems = maxItems,
                description = description,
                nullable = nullable,
            )
            Kind.STRING -> StringSchema(
                format = format,
                pattern = pattern,
                minLength = minLength,
                maxLength = maxLength,
                enumValues = enumValues?.toList(),
                example = example,
                description = description,
                nullable = nullable,
            )
            Kind.INTEGER -> IntegerSchema(
                format = format,
                minValue = minValue?.toLong(),
                maxValue = maxValue?.toLong(),
                description = description,
                nullable = nullable,
            )
            Kind.NUMBER -> NumberSchema(
                format = format,
                minValue = minValue,
                maxValue = maxValue,
                description = description,
                nullable = nullable,
            )
            Kind.BOOLEAN -> BooleanSchema(
                description = description,
                nullable = nullable,
            )
            null -> error("JsonSchemaBuilder: no type set and no anyOf variants — nothing to build")
        }
    }
}
