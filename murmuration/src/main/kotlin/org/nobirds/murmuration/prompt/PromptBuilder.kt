package org.nobirds.murmuration.prompt

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.llm.*

import kotlinx.serialization.json.Json
import kotlinx.serialization.serializer

interface PromptContentBuilder {
    fun part(part: PromptPart)
    fun build(): PromptContent

    companion object {
        operator fun invoke(type: PromptContentType): PromptContentBuilder = PromptContentBuilderImpl(type)
    }

    private class PromptContentBuilderImpl(private val type: PromptContentType) : PromptContentBuilder {
        private val parts = mutableListOf<PromptPart>()

        override fun part(part: PromptPart) {
            parts.add(part)
        }

        override fun build(): PromptContent = PromptContent(type, parts)
    }
}

interface PromptBuilder {
    var llm: Llm?
    var settings: PromptSettings?

    fun content(content: List<PromptContent>)
    fun content(content: PromptContent)
    fun build(): Prompt

    companion object {
        operator fun invoke(): PromptBuilder = PromptBuilderImpl()
    }

    private class PromptBuilderImpl : PromptBuilder {
        override var llm: Llm? = null
        override var settings: PromptSettings? = null
        private val contents = mutableListOf<PromptContent>()

        override fun content(content: List<PromptContent>) {
            contents.addAll(content)
        }

        override fun content(content: PromptContent) {
            contents.add(content)
        }

        override fun build(): Prompt = Prompt(
            llm ?: error("LLM must be specified"),
            settings,
            contents.toList()
        )
    }
}


operator fun Prompt.plus(content: PromptContent): Prompt = update { content(content) }
operator fun Prompt.plus(content: List<PromptContent>): Prompt = update { content(content) }

fun Prompt.toBuilder(): PromptBuilder = PromptBuilder().apply {
    val p = this@toBuilder
    this.llm = p.llm
    this.settings = p.settings
    this.content(p.content)
}

inline fun Prompt.update(block: PromptBuilder.(Prompt) -> Unit): Prompt = toBuilder().apply {
    block(this@update)
}.build()

fun interface PromptGenerator<In> {
    suspend fun ExecutionContext.generate(input: In): Prompt

    companion object {
        inline fun <In> build(crossinline builder: suspend PromptBuilder.(ExecutionContext, In) -> Unit): PromptGenerator<In> = PromptGenerator { input ->
            val b = PromptBuilder()
            b.builder(this, input)
            b.build()
        }
        inline fun <In> build(llm: Llm, crossinline builder: suspend PromptBuilder.(ExecutionContext, In) -> Unit): PromptGenerator<In> = PromptGenerator { input ->
            val b = PromptBuilder()
            b.llm = llm
            b.builder(this, input)
            b.build()
        }
    }
}


// DSL Extensions for nicer syntax

inline fun buildPrompt(block: PromptBuilder.() -> Unit): Prompt = PromptBuilder().apply(block).build()

inline fun PromptBuilder.system(block: PromptContentBuilder.() -> Unit) {
    val builder = PromptContentBuilder(PromptContentType.SYSTEM)
    builder.block()
    content(builder.build())
}

inline fun PromptBuilder.user(block: PromptContentBuilder.() -> Unit) {
    val builder = PromptContentBuilder(PromptContentType.USER)
    builder.block()
    content(builder.build())
}

inline fun PromptBuilder.model(block: PromptContentBuilder.() -> Unit) {
    val builder = PromptContentBuilder(PromptContentType.MODEL)
    builder.block()
    content(builder.build())
}


fun PromptContentBuilder.text(text: String, label: String? = null, cacheStrategy: PromptCacheStrategy = PromptCacheStrategy.NO_CACHE, replaceableSectionId: String? = null) {
    val content = if (label != null) "$label\n$text" else text
    part(TextPromptPart(content, cacheStrategy = cacheStrategy, replaceableSectionId = replaceableSectionId))
}

inline fun PromptContentBuilder.text(label: String? = null, cacheStrategy: PromptCacheStrategy = PromptCacheStrategy.NO_CACHE, replaceableSectionId: String? = null, block: StringBuilder.() -> Unit) {
    text(buildString(block), label, cacheStrategy, replaceableSectionId)
}

object PromptSection {
    const val CURRENT_OUTPUT_SNAPSHOT = "current-output-snapshot"
}

fun PromptContentBuilder.image(bytes: ByteArray, mimeType: String, label: String? = null, name: String? = null, cacheStrategy: PromptCacheStrategy = PromptCacheStrategy.NO_CACHE) {
    if (label != null) text(label, cacheStrategy = cacheStrategy)
    part(DataPromptPart(bytes, mimeType, name, cacheStrategy = cacheStrategy))
}

fun PromptContentBuilder.fileUri(uri: String, mimeType: String? = null, label: String? = null, name: String? = null, cacheStrategy: PromptCacheStrategy = PromptCacheStrategy.NO_CACHE) {
    if (label != null) text(label, cacheStrategy = cacheStrategy)
    part(FileDataPromptPart(uri, name, mimeType, cacheStrategy = cacheStrategy))
}

/**
 * JSON used by [json] / `jsonWithSchema` to serialize a value INTO the prompt shown to the model.
 *
 * encodeDefaults=true is mandatory for LLM-facing output: with the kotlinx default (false) a field
 * sitting at its default value (`count = 1`, `enabled = false`) is dropped from the rendered JSON,
 * so the model cannot tell "default value" apart from "field absent / empty". That ambiguity drove a
 * self-review round-trip loop in application snapshots. Prompt renders
 * must show defaults explicitly; only genuinely empty state is elided.
 */
@PublishedApi
internal val defaultJson = Json {
    ignoreUnknownKeys = true
    encodeDefaults = true
}

inline fun <reified T : Any> PromptContentBuilder.json(value: T, label: String? = null, cacheStrategy: PromptCacheStrategy = PromptCacheStrategy.NO_CACHE) {
    if (label != null) {
        text(label, cacheStrategy = cacheStrategy)
    }
    part(TextPromptPart(defaultJson.encodeToString(serializer<T>(), value), cacheStrategy = cacheStrategy))
}

val PromptContentBuilder.PIPELINE get() = PromptCacheStrategy.PIPELINE
val PromptContentBuilder.NO_CACHE get() = PromptCacheStrategy.NO_CACHE
fun PromptContentBuilder.TTL(duration: kotlin.time.Duration) = PromptCacheStrategy.ttl(duration)
