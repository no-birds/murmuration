package org.nobirds.murmuration.anthropic

import com.anthropic.models.messages.JsonOutputFormat
import com.anthropic.models.messages.OutputConfig
import com.anthropic.models.messages.Tool
import kotlinx.serialization.InternalSerializationApi
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.serializer
import org.nobirds.murmuration.prompt.InvalidRequestException
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.TypedPromptSettings
import org.nobirds.murmuration.schema.FunctionSchema
import org.nobirds.murmuration.schema.JsonSchema
import org.nobirds.murmuration.schema.toJsonSchema
import org.nobirds.murmuration.schema.toStandardJson

/*
 * Mapping of tool schemas and structured-output schemas to the Messages API types. Pure functions: the request itself
 * is assembled elsewhere. Tool choice is not mapped on purpose: the abstraction has no forced tool call, so the
 * provider sends nothing (= `auto`).
 *
 * The schema is emitted by the core standard-JSON-Schema emitter (toStandardJson); this file only moves it into SDK
 * types and, where the API is strict, checks it against AnthropicSchemaSubset before the network.
 */

/**
 * Maps this function to an SDK [Tool] whose `input_schema` is the standard JSON Schema of the parameters.
 *
 * @param strict `true` sends the tool as a strict one: the schema must fit [AnthropicSchemaSubset] (else
 * [InvalidRequestException]) and every object is closed. `false` (what the product uses) sends the schema as declared:
 * the API rejects constraints like `minimum` only for strict schemas.
 */
internal fun FunctionSchema.toAnthropicTool(prompt: Prompt, strict: Boolean = false): Tool {
    if (strict) parameters.requireInSubset(prompt, "tool '$name'")
    val schema = parameters.toStandardJson(closedObjects = strict)
    val inputSchema = Tool.InputSchema.builder()
    (schema["properties"] as? JsonObject)?.let { properties ->
        inputSchema.properties(
            Tool.InputSchema.Properties.builder()
                .additionalProperties(properties.mapValues { it.value.toJsonValue() })
                .build()
        )
    }
    (schema["required"] as? JsonArray)?.let { required ->
        inputSchema.required(required.map { (it as JsonPrimitive).content })
    }
    schema.filterKeys { it != "type" && it != "properties" && it != "required" }
        .forEach { (key, value) -> inputSchema.putAdditionalProperty(key, value.toJsonValue()) }
    return Tool.builder()
        .name(name)
        .description(description)
        .inputSchema(inputSchema.build())
        .apply { if (strict) strict(true) }
        .build()
}

/**
 * Maps this schema to `output_config.format = json_schema` with all objects closed. A schema outside
 * [AnthropicSchemaSubset] is [InvalidRequestException] before the network.
 */
internal fun JsonSchema.toAnthropicOutputConfig(prompt: Prompt): OutputConfig {
    requireInSubset(prompt, "structured output")
    val format = JsonOutputFormat.Schema.builder()
    toStandardJson(closedObjects = true).forEach { (key, value) -> format.putAdditionalProperty(key, value.toJsonValue()) }
    return OutputConfig.builder().format(JsonOutputFormat.of(format.build())).build()
}

/** Structured output for [TypedPromptSettings]: the schema of [TypedPromptSettings.type] as `output_config.format`. */
@OptIn(InternalSerializationApi::class)
internal fun TypedPromptSettings<*>.toAnthropicOutputConfig(prompt: Prompt): OutputConfig =
    type.serializer().toJsonSchema().toAnthropicOutputConfig(prompt)

private fun JsonSchema.requireInSubset(prompt: Prompt, what: String) {
    val violations = AnthropicSchemaSubset.violations(this)
    if (violations.isNotEmpty()) {
        throw InvalidRequestException(
            prompt, httpCode = null,
            message = "Schema of $what is outside the JSON Schema subset Claude supports: ${violations.joinToString("; ")}",
            cause = null,
        )
    }
}
