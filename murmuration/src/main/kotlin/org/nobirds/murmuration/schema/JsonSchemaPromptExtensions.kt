package org.nobirds.murmuration.schema

import kotlinx.serialization.serializer
import org.nobirds.murmuration.prompt.PromptCacheStrategy
import org.nobirds.murmuration.prompt.PromptContentBuilder
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.json
import org.nobirds.murmuration.prompt.text

/**
 * Adds the input schema for [T] to the prompt as a JSON Schema string, so the model can see
 * the expected shape. Useful right before `json(value)` for tool / structured input.
 */
inline fun <reified T : JsonSchemable> PromptContentBuilder.inputSchema(
    cacheStrategy: PromptCacheStrategy = PromptCacheStrategy.PIPELINE,
) {
    // T : JsonSchemable marks types shown to the model as a schema. The bound makes their set
    // reflectively enumerable (a scan-test builds a strict schema for every implementor), and the
    // schema itself is built strictly: an empty object / Map / under-annotated type fails on CI
    // rather than at agent setup in prod.
    val schemaString = serializer<T>().toJsonSchema().toJsonString()
    part(TextPromptPart(schemaString, cacheStrategy = cacheStrategy))
}

/**
 * Shortcut for adding both a value of type [T] and its JSON Schema to the prompt. Handy for
 * structured input where the model needs both the format description and a concrete
 * example. Under the hood: [inputSchema] + [json].
 *
 * Separate strategies for schema and data: the schema is a **constant** per pipeline (generated
 * from a data class and does not change between round-trips), data is a mutable snapshot. So
 * `schemaCacheStrategy` defaults to `PIPELINE` (gets into the cache and is reused between invocations
 * of the same agent), and `valueCacheStrategy` is `NO_CACHE`. Before the separate strategies the whole pair
 * went as NO_CACHE, and the schema (~9K tokens per jsonWithSchema call) went every time as
 * regular_input — noticeably more expensive than cached_input.
 *
 * The label (if set) is also a constant, goes via `schemaCacheStrategy`.
 *
 * For this to work, the provider must use the `GroupByCacheability` strategy,
 * otherwise with `Strict` the schema, placed in one USER message after NO_CACHE parts, will not
 * get into the cache prefix.
 */
inline fun <reified T : JsonSchemable> PromptContentBuilder.jsonWithSchema(
    value: T,
    label: String? = null,
    schemaCacheStrategy: PromptCacheStrategy = PromptCacheStrategy.PIPELINE,
    valueCacheStrategy: PromptCacheStrategy = PromptCacheStrategy.NO_CACHE,
) {
    if (label != null) text(label, cacheStrategy = schemaCacheStrategy)
    inputSchema<T>(cacheStrategy = schemaCacheStrategy)
    json(value, cacheStrategy = valueCacheStrategy)
}

/**
 * List variant: shows the array [values] together with an `array<T>` schema. Needed because a
 * `List<T>` does not itself carry the [JsonSchemable] marker (the element type [T] does), while
 * the scalar [jsonWithSchema] bound requires it. The array schema is built from
 * `serializer<List<T>>()`.
 */
inline fun <reified T : JsonSchemable> PromptContentBuilder.jsonWithSchema(
    values: List<T>,
    label: String? = null,
    schemaCacheStrategy: PromptCacheStrategy = PromptCacheStrategy.PIPELINE,
    valueCacheStrategy: PromptCacheStrategy = PromptCacheStrategy.NO_CACHE,
) {
    if (label != null) text(label, cacheStrategy = schemaCacheStrategy)
    val schemaString = serializer<List<T>>().toJsonSchema().toJsonString()
    part(TextPromptPart(schemaString, cacheStrategy = schemaCacheStrategy))
    json(values, cacheStrategy = valueCacheStrategy)
}
