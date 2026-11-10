@file:OptIn(ExperimentalSerializationApi::class)

package org.nobirds.murmuration.schema

import kotlinx.serialization.ExperimentalSerializationApi
import org.nobirds.murmuration.prompt.InstrumentDeclaration
import org.nobirds.murmuration.prompt.InstrumentParameter

/**
 * Provider-neutral representation of a tool/function schema. Carries everything a
 * function-calling provider needs to expose the tool to an LLM: the name, the
 * human-readable description, and a fully-built [JsonSchema] for the parameters object
 * (including required / descriptions / validator-projected constraints).
 *
 * Provider modules turn this into their SDK-specific type with a pure mechanical mapping —
 * they do not read annotations, resolve validators, or assemble the parameter object
 * themselves; that all happens in [InstrumentDeclaration.toFunctionSchema].
 */
data class FunctionSchema(
    val name: String,
    val description: String,
    /** Always an [ObjectSchema] in practice — `parameters` of a function are an object. */
    val parameters: JsonSchema,
)

/**
 * Builds the provider-neutral [FunctionSchema] for an instrument: assembles the parameter
 * object schema by walking each parameter's serializer with its parameter-level
 * annotations, attaches per-parameter descriptions, and marks non-optional parameters as
 * required. All annotation reading, [ParameterValidator] resolution and composing happen
 * here; provider modules get a finished [JsonSchema] tree.
 */
fun InstrumentDeclaration<*>.toFunctionSchema(): FunctionSchema {
    val parametersSchema = jsonSchema {
        type(JsonSchemaBuilder.Kind.OBJECT)
        for (param in parameters) {
            val paramSchema = param.type.parameterSerializer().toJsonSchema(param.annotations)

            // anyOf-trees can't carry a top-level description in some providers (notably
            // Gemini puts it on the parent, not on each variant), so we leave them as-is.
            val withDescription = if (param.description.isNotBlank() && paramSchema !is AnyOfSchema)
                paramSchema.withDescription(param.descriptionOver(paramSchema))
            else paramSchema

            property(param.name, withDescription)
            if (!param.isOptional) {
                required(param.name)
            }
        }
    }
    return FunctionSchema(
        name = name,
        description = description,
        parameters = parametersSchema,
    )
}

/**
 * The description to put on a parameter node: the parameter's own, followed by the legend that the enum class
 * carries in its class-level `@SchemaInfo` (already on [schema] by the walker) — the legend explains the short
 * serial names, so a parameter description must not shadow it.
 */
private fun InstrumentParameter.descriptionOver(schema: JsonSchema): String {
    val legend = (schema as? StringSchema)?.takeIf { it.enumValues != null }?.description?.takeIf { it.isNotBlank() }
    return if (legend == null || legend == description) description else "$description $legend"
}

/**
 * Returns a copy of the schema with [description] set, dispatching to the right copy()
 * for each [JsonSchema] subtype. Used by [toFunctionSchema] to attach parameter-level
 * descriptions to the per-parameter schema node.
 */
private fun JsonSchema.withDescription(description: String): JsonSchema = when (this) {
    is ObjectSchema -> copy(description = description)
    is ArraySchema -> copy(description = description)
    is StringSchema -> copy(description = description)
    is IntegerSchema -> copy(description = description)
    is NumberSchema -> copy(description = description)
    is BooleanSchema -> copy(description = description)
    is AnyOfSchema -> copy(description = description)
}
