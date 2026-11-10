package org.nobirds.murmuration.prompt

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.llm.*

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement
import kotlin.reflect.KClass
import kotlin.reflect.KType
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

data class InstrumentParameter(
    val name: String,
    val type: KType,
    val description: String = "",
    val isOptional: Boolean = false,
    /**
     * All parameter annotations (including those inherited from same-named parameters in
     * super-classes, when the factory merges them in). Used by `InstrumentDecorator`s and
     * `ParameterValidator`s.
     */
    val annotations: List<Annotation> = emptyList(),
)

interface InstrumentDeclaration<out R> : InstrumentsDeclaration<Any> {
    val name: String
    val description: String
    val parameters: List<InstrumentParameter>
    val returnType: KType
    /**
     * All annotations attached to the instrument (typically function-level, plus those
     * inherited from same-named methods in super-classes). Used by `InstrumentDecorator`s,
     * which let user code implement custom contracts without caring about the instrument's
     * origin.
     */
    val annotations: List<Annotation> get() = emptyList()

    /**
     * Converts the provider's raw JSON arguments into the values accepted by [invoke].
     * The default is the established [JsonElement.toAny] conversion. A declaration that
     * needs some parameters as exact [JsonElement]s overrides this, and every decorator
     * that wraps a declaration must forward to its delegate so wrapping keeps the behaviour.
     */
    fun prepareWireArguments(args: Map<String, JsonElement>): Map<String, Any?> =
        args.mapValues { it.value.toAny() }

    suspend fun invoke(args: Map<String, Any?>): R

    override val instruments: List<InstrumentDeclaration<*>> get() = listOf(this)
}

interface InstrumentsDeclaration<out T : Any> {
    val obj: T
    val instruments: List<InstrumentDeclaration<*>>

    companion object {
        fun <T: Any> of(obj: T, instruments: List<InstrumentDeclaration<*>>): InstrumentsDeclaration<T> =
            InstrumentsDeclarationImpl(obj, instruments)
    }
}

private data class InstrumentsDeclarationImpl<out T : Any>(
    override val obj: T,
    override val instruments: List<InstrumentDeclaration<*>>,
) : InstrumentsDeclaration<T>

interface PromptSettings {
    /**
     * Whether to ask the provider to return chain-of-thought in the response (for thinking models).
     * Thinking-token billing ALWAYS happens (thoughts are computed on the model side); the flag
     * only controls whether the trace is returned — the DB does not bloat, the network response is shorter. Defaults
     * to `true` for backward compatibility. Overridden for a pipeline via
     * [PromptExecutor.withIncludeThoughts].
     */
    val includeThoughts: Boolean get() = true

    /**
     * Per-prompt no-chunk streaming timeout. `null` — use the global default from the executor
     * config. Useful for image agents, where Imagen sometimes normally takes 130-160s — on a
     * global 120s limit such requests are cut off at the edge with `finish_reason=UNSPECIFIED`.
     * This is a **client-side control**: the executor handles it locally, it is not sent to the LLM API.
     */
    val streamIdleTimeout: Duration? get() = null

    /**
     * Per-prompt thinking budget (provider-config). `null` — the model's default behaviour.
     * Passed to Gemini as `thinkingConfig.thinkingBudget`. Positive values are a
     * cap on thinking tokens, `0` — disable thinking, `-1` — unlimited. Useful
     * for strict finalize/check phases, where the model spinning up to "think about the architecture"
     * is a waste of money.
     */
    val thinkingBudget: Int? get() = null

    /**
     * Provider-neutral thinking level (LOW/MEDIUM/HIGH). Mutually exclusive with
     * [thinkingBudget] for models that only accept one of them (e.g. Gemini 3 Pro uses
     * `thinkingLevel`, Gemini 2.5 Flash uses `thinkingBudget`). When both are set the
     * provider mapping picks the one its API supports; for Gemini that means
     * `thinkingLevel` wins for Pro-class models, `thinkingBudget` for Flash.
     */
    val thinkingLevel: ThinkingLevel? get() = null

    /**
     * Per-prompt cap on output tokens (provider-config). `null` — let the executor apply
     * its global default. Mapped to Gemini's `maxOutputTokens`. The cap covers content
     * tokens AND thinking tokens combined — provider enforces it as a hard ceiling and
     * stops generation with `finish_reason=MAX_TOKENS`.
     *
     * Use to bound runaway-loop blast radius (a misbehaving model that spams identical
     * tool-calls hits the cap an order of magnitude faster than the model default, MAX_TOKENS
     * surfaces as [MaxTokensException], recovery kicks in instead of silently aggregating
     * 1000+ tool calls into pipeline state) and to keep cost predictable.
     */
    val maxOutputTokens: Int? get() = null

}

data class TypedPromptSettings<T : Any>(
    val type: KClass<T>,
    override val includeThoughts: Boolean = true,
    override val streamIdleTimeout: Duration? = null,
    override val thinkingBudget: Int? = null,
    override val thinkingLevel: ThinkingLevel? = null,
    override val maxOutputTokens: Int? = null,
) : PromptSettings
data class ToolsPromptSettings(
    val tools: List<InstrumentsDeclaration<*>>,
    override val includeThoughts: Boolean = true,
    override val streamIdleTimeout: Duration? = null,
    override val thinkingBudget: Int? = null,
    override val thinkingLevel: ThinkingLevel? = null,
    override val maxOutputTokens: Int? = null,
) : PromptSettings

/**
 * Settings for image generation. Signals to the executor that the prompt goes to an Imagen model,
 * and carries the required image size. Unifies the image path under the common `PromptExecutor`:
 * images arrive as `InlineData` events in the common Flow, no separate interface is needed.
 *
 * `includeThoughts = false` by default — Imagen does not generate chain-of-thought.
 *
 * `streamIdleTimeout = 180.seconds` by default — Imagen regularly takes 130-160s in
 * normal successful requests (in prod we saw SUCCESS at 113s, 165s and FAIL UNSPECIFIED at
 * 121s/132s — on the edge of the general 120s limit). The raised default gives headroom so that
 * successful requests are not cut off. Text prompts stay on the global default from the executor config.
 */
data class ImagePromptSettings(
    val size: ImageSize = ImageSize.DEFAULT,
    override val includeThoughts: Boolean = false,
    override val streamIdleTimeout: Duration? = 180.seconds,
    override val thinkingBudget: Int? = null,
    override val thinkingLevel: ThinkingLevel? = null,
    override val maxOutputTokens: Int? = null,
) : PromptSettings

/**
 * Settings for a text-embedding operation. Signals the executor to route the prompt to an
 * embedding model (`embedContent`) instead of `generateContent` — analogous to how
 * [ImagePromptSettings] signals an image model. The prompt's user text is what gets embedded;
 * [outputDimensionality] and [taskType] configure the embedding model.
 *
 * Thinking / tools / output-token settings don't apply to embeddings — [includeThoughts] is
 * forced false and the rest stay at their no-op defaults.
 */
data class EmbeddingPromptSettings(
    val outputDimensionality: Int = 768,
    val taskType: String = "SEMANTIC_SIMILARITY",
) : PromptSettings {
    override val includeThoughts: Boolean = false
}

/**
 * Settings for a music-generation operation (e.g. Lyria 3 on Vertex). Music generation rides the
 * SAME `generateContent` round-trip as text and image — the model returns audio as an `InlineData`
 * part (`audio/mpeg`) that a `BinaryDataStorage` persists to a file, exactly like image output.
 * Signalled here so the executor requests the AUDIO response modality (analogous to how
 * [ImagePromptSettings] requests IMAGE).
 *
 * `includeThoughts = false` — Lyria emits no chain-of-thought. `streamIdleTimeout = 300.seconds`
 * by default — Lyria 3 Pro renders up to ~3-minute tracks and can take a while; the global 120s
 * default would cut otherwise-successful generations.
 */
data class MusicPromptSettings(
    override val includeThoughts: Boolean = false,
    override val streamIdleTimeout: Duration? = 300.seconds,
    override val thinkingBudget: Int? = null,
    override val thinkingLevel: ThinkingLevel? = null,
    override val maxOutputTokens: Int? = null,
) : PromptSettings

interface Prompt {
    val llm: Llm
    val settings: PromptSettings?
    val content: List<PromptContent>

    companion object {
        operator fun invoke(llm: Llm, settings: PromptSettings?, content: List<PromptContent>): Prompt = PromptImpl(llm, settings, content)
    }

    private data class PromptImpl(override val llm: Llm, override val settings: PromptSettings?, override val content: List<PromptContent>) : Prompt
}

@Serializable
enum class PromptContentType {
    SYSTEM,
    USER,
    MODEL,
    TOOL,
}

enum class PromptCacheScope {
    NO_CACHE,
    PIPELINE,
    TTL_BASED
}

data class PromptCacheStrategy(
    val scope: PromptCacheScope,
    val ttl: Duration? = null
) {
    companion object {
        val NO_CACHE = PromptCacheStrategy(PromptCacheScope.NO_CACHE)
        val PIPELINE = PromptCacheStrategy(PromptCacheScope.PIPELINE)
        fun ttl(ttl: Duration) = PromptCacheStrategy(PromptCacheScope.TTL_BASED, ttl)
    }
}

sealed interface PromptPart {
    val cacheStrategy: PromptCacheStrategy
}

data class DataPromptPart(
    val data: ByteArray,
    val mimeType: String,
    val name: String? = null,
    override val cacheStrategy: PromptCacheStrategy = PromptCacheStrategy.NO_CACHE
) : PromptPart

data class FileDataPromptPart(
    val uri: String,
    val name: String? = null,
    val mimeType: String? = null,
    override val cacheStrategy: PromptCacheStrategy = PromptCacheStrategy.NO_CACHE
) : PromptPart

/**
 * Kind of a [TextPromptPart]: ordinary text, or a model "thought" (reasoning) part. The kind is
 * set explicitly by whoever builds the part (the agent, from the response event type); it is
 * never inferred from the presence of a signature.
 */
enum class PartKind { TEXT, THOUGHT }

data class TextPromptPart(
    val text: String,
    val thoughtSignature: String? = null,
    override val cacheStrategy: PromptCacheStrategy = PromptCacheStrategy.NO_CACHE,
    val replaceableSectionId: String? = null,
    val kind: PartKind = PartKind.TEXT,
) : PromptPart

data class ToolCallPromptPart(
    val call: ToolCall,
    override val cacheStrategy: PromptCacheStrategy = PromptCacheStrategy.NO_CACHE,
    val replaceableSectionId: String? = null,
) : PromptPart

data class ToolResultPromptPart(
    val name: String,
    val result: JsonElement,
    val callId: CallId,
    val thoughtSignature: String? = null,
    override val cacheStrategy: PromptCacheStrategy = PromptCacheStrategy.NO_CACHE,
    val replaceableSectionId: String? = null,
) : PromptPart

/** Universal accessor for the optional pruning section id across part types. */
fun PromptPart.replaceableSectionId(): String? = when (this) {
    is TextPromptPart -> replaceableSectionId
    is ToolCallPromptPart -> replaceableSectionId
    is ToolResultPromptPart -> replaceableSectionId
    is DataPromptPart, is FileDataPromptPart -> null
}

interface PromptContent {
    val type: PromptContentType
    val parts: List<PromptPart>

    companion object {
        operator fun invoke(type: PromptContentType, parts: List<PromptPart>): PromptContent = PromptContentImpl(type, parts)
    }

    private data class PromptContentImpl(override val type: PromptContentType, override val parts: List<PromptPart>) : PromptContent
}
