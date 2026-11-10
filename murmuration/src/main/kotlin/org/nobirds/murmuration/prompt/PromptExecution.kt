package org.nobirds.murmuration.prompt

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.llm.*

import kotlinx.coroutines.flow.Flow
import kotlinx.serialization.KSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import org.nobirds.murmuration.internal.UUIDSerializer
import java.util.UUID

/**
 * Input token breakdown: prompt tokens, how many of them were read from cache, and how many were
 * written to cache (billed separately, by TTL class). Providers without cache-write billing report 0;
 * the write fields have no constructor default, so a forgotten class is a compile error. A dump written
 * before they existed is read through [InputTokenStatisticsSerializer], where a missing field is 0.
 */
@Serializable(with = InputTokenStatisticsSerializer::class)
data class InputTokenStatistics(
    val total: Long,
    val cached: Long,
    val cacheWriteShort: Long,
    val cacheWriteLong: Long
) {
    /** Input billed at the regular price: total − cached − cache writes, never negative. */
    val uncached: Long get() = (total - cached - cacheWriteShort - cacheWriteLong).coerceAtLeast(0)

    override fun toString(): String = buildString {
        append(total)
        if (cached > 0) append(" (cached: ").append(cached).append(')')
        if (cacheWriteShort > 0) append(" (cache write 5m: ").append(cacheWriteShort).append(')')
        if (cacheWriteLong > 0) append(" (cache write 1h: ").append(cacheWriteLong).append(')')
    }

    companion object {
        val EMPTY = InputTokenStatistics(0, 0, 0, 0)
    }
}

operator fun InputTokenStatistics.plus(other: InputTokenStatistics): InputTokenStatistics =
    InputTokenStatistics(
        total + other.total,
        cached + other.cached,
        cacheWriteShort + other.cacheWriteShort,
        cacheWriteLong + other.cacheWriteLong
    )

operator fun InputTokenStatistics.minus(other: InputTokenStatistics): InputTokenStatistics =
    InputTokenStatistics(
        total - other.total,
        cached - other.cached,
        cacheWriteShort - other.cacheWriteShort,
        cacheWriteLong - other.cacheWriteLong
    )

/**
 * Output token breakdown: completion tokens and how many were used for thinking.
 */
@Serializable
data class OutputTokenStatistics(
    val total: Long = 0,
    val thinking: Long = 0
) {
    override fun toString(): String = if (thinking > 0) "$total (thinking: $thinking)" else "$total"

    companion object {
        val EMPTY = OutputTokenStatistics()
    }
}

operator fun OutputTokenStatistics.plus(other: OutputTokenStatistics): OutputTokenStatistics =
    OutputTokenStatistics(total + other.total, thinking + other.thinking)

operator fun OutputTokenStatistics.minus(other: OutputTokenStatistics): OutputTokenStatistics =
    OutputTokenStatistics(total - other.total, thinking - other.thinking)
/**
 * Full token usage for a single prompt execution, split into input and output.
 */
@Serializable
data class PromptExecutionTokenStatistics(
    val totalTokens: Long = 0,
    val input: InputTokenStatistics = InputTokenStatistics.EMPTY,
    val output: OutputTokenStatistics = OutputTokenStatistics()
)

/**
 * Statistics for a single prompt execution.
 *
 * [modelName] is the actual model name returned by the API (for example
 * `gemini-2.5-flash-preview-04-17`, with the date version suffix). [LlmCostCalculator]
 * resolves the matching [Llm] through [LlmRegistry] and reads `pricingProfile` from it.
 *
 * [requestedModelName] is the name we asked for (`prompt.llm.model`). Nullable for
 * backwards compatibility with previously serialised statistics; new records always fill
 * it. The pair `(requestedModelName, modelName)` is meant for divergence analytics
 * (what we asked for vs. what came back) and for spotting matcher drift.
 */
@Serializable
data class PromptExecutionStatistics(
    val modelName: String = "",
    val requestedModelName: String? = null,
    val attempts: Int = 1,
    val success: Boolean = true,
    val errorMessage: String? = null,
    val tokens: PromptExecutionTokenStatistics = PromptExecutionTokenStatistics(),
    /**
     * Provider-neutral dump of what was sent to the model: system + user/model history,
     * each as a [PromptMessage] of [MessagePart]s. Filled by the executor; consumers
     * (statistics persistence, admin tooling) read this directly instead of parsing
     * provider-specific payloads.
     */
    val requestMessages: List<PromptMessage> = emptyList(),
    /**
     * Provider-neutral dump of what came back from the model — a single message with
     * role = `MODEL` (text + tool calls + inline data, etc.). `null` when the call
     * failed before producing a message.
     */
    val responseMessage: PromptMessage? = null,
    val cacheId: String? = null,
    val cachedTokenCount: Long? = null,
    val isCacheCreated: Boolean? = null,
    val cacheTtlMillis: Long? = null,
    val isCacheExpired: Boolean? = null,
    val cacheExpiredAtMillis: Long? = null,
    val cacheScope: PromptCacheScope? = null,
    /**
     * Formal sub-classification of the failure (typed exception's enum value, e.g.
     * `"PROHIBITED"`, `"NETWORK_TIMEOUT"`, `"CACHE_EXPIRED"`). `null` for SUCCESS or for
     * untyped failures (raw [Exception] without a [PromptExecutionException] mapping).
     * Populated in agent catch-blocks via [PromptExecutionException.errorCode].
     */
    val errorCode: String? = null,
    /**
     * Provider's raw `finish_reason` token if the failure carries one (currently only
     * `PolicyViolationException`, e.g. `"IMAGE_PROHIBITED_CONTENT"`). `null` otherwise.
     */
    val finishReason: String? = null,
    /**
     * Per-attempt streaming diagnostics: how many chunks arrived, when the last one came,
     * what kinds of parts were inside (text/thought/tool_call), and the last chunk-level
     * finish_reason. Populated by streaming executors (e.g. Gemini SSE). `null` for
     * direct/non-streaming responses or for failures that happen before the stream opens.
     *
     * Used by analytics to investigate stream truncations like «Provider ended stream
     * without a finish reason» — typical pattern: many chunks then a large `sinceLastChunkMs`
     * gap before close = server-side cap; zero chunks = connect failure; mid-stream drop
     * with small `sinceLastChunkMs` = transport blip.
     */
    val streamDiagnostics: StreamDiagnostics? = null,
)

/**
 * Snapshot of a streaming execution for post-mortem analytics. All durations in
 * milliseconds; `partsX` are counts of how many of that kind of Part arrived across
 * all chunks in the attempt.
 */
@Serializable
data class StreamDiagnostics(
    val chunkCount: Int,
    val totalElapsedMs: Long,
    val sinceLastChunkMs: Long,
    val partsText: Int,
    val partsThought: Int,
    val partsToolCall: Int,
    /** finish_reason from the last received chunk; `null` if no chunk carried it. */
    val lastChunkFinishReason: String? = null,
    /**
     * Time to wait for the FIRST chunk from the start of the stream (TTFB). `null` if no chunk
     * arrived (a break before the first event — e.g. a TTFB timeout). Previously it was computed only
     * inside `if (logger.isDebugEnabled)` for the log line and was stored nowhere — with DEBUG
     * off the value did not exist at all. See [StreamTimingTracker].
     */
    val firstChunkMs: Long? = null,
    /**
     * The largest gap between two CONSECUTIVE chunks over the whole stream. `null` if
     * fewer than two chunks arrived (no pair to compute a gap from). See [StreamTimingTracker].
     */
    val maxGapMs: Long? = null,
)

/**
 * Tracks per-chunk stream timing for [StreamDiagnostics]: time-to-first-chunk ([firstChunkMs])
 * and the largest gap between two consecutive chunks ([maxGapMs]). Pure state, no I/O — driven by
 * the caller feeding it each chunk's arrival timestamp (`System.nanoTime()`), so it is
 * unit-testable without a fake provider stream (the genai `Client`/`ResponseStream` are not
 * mockable — see `GeminiChunkFlowTimeoutTest`'s comment). Shared between `murmuration-gemini` and
 * `murmuration-openai` so the timing math is not duplicated a second time (the OpenAI side already
 * carried one exact copy of `buildStreamDiag` labelled "copy of the gemini helper").
 */
class StreamTimingTracker(private val streamStartedAt: Long) {
    private var firstChunkAt: Long? = null
    private var lastChunkAt: Long = streamStartedAt
    private var maxGapNanos: Long? = null

    /** Records a chunk's arrival at [now] (`System.nanoTime()`). Call once per chunk, in order. */
    fun recordChunk(now: Long) {
        val first = firstChunkAt
        if (first == null) {
            firstChunkAt = now
        } else {
            val gap = now - lastChunkAt
            maxGapNanos = maxOf(maxGapNanos ?: 0L, gap)
        }
        lastChunkAt = now
    }

    /** `null` until the first chunk has been recorded. */
    val firstChunkMs: Long? get() = firstChunkAt?.let { (it - streamStartedAt) / 1_000_000 }

    /** `null` until at least two chunks have been recorded. */
    val maxGapMs: Long? get() = maxGapNanos?.let { it / 1_000_000 }

    /**
     * Snapshots the streaming-diagnostics counters at the current instant — the ONE construction of
     * [StreamDiagnostics]: `murmuration-gemini` and
     * `murmuration-openai` each carried an identical private `buildStreamDiag` taking
     * `streamStartedAt`/`lastChunkAt` as separate params even though this tracker already holds
     * both internally (its constructor arg and the timestamp its own [recordChunk] just stored) —
     * a field added to [StreamDiagnostics] later now reaches both providers by construction, not by
     * the discipline of editing two files. [chunkCount]/[totalTextParts]/[totalThoughtParts]/
     * [totalToolCallParts]/[lastFinishReason] stay provider-side counters (Gemini and OpenAI decode
     * them from different SDK shapes) — only the timing/assembly half moves here.
     */
    fun snapshot(
        chunkCount: Int,
        totalTextParts: Int,
        totalThoughtParts: Int,
        totalToolCallParts: Int,
        lastFinishReason: String?,
    ): StreamDiagnostics {
        val now = System.nanoTime()
        return StreamDiagnostics(
            chunkCount = chunkCount,
            totalElapsedMs = (now - streamStartedAt) / 1_000_000,
            sinceLastChunkMs = (now - lastChunkAt) / 1_000_000,
            partsText = totalTextParts,
            partsThought = totalThoughtParts,
            partsToolCall = totalToolCallParts,
            lastChunkFinishReason = lastFinishReason,
            firstChunkMs = firstChunkMs,
            maxGapMs = maxGapMs,
        )
    }
}

sealed interface PromptExecutionResult

/**
 * One text Part of the model response as it came from the API. Preserves the fact "this is chain-of-thought
 * (a model thought) or a regular response" via `isThought`, plus `thoughtSignature` for
 * follow-up requests — no information is dropped at the `fromGenai` stage.
 */
@Serializable
data class ExecutionTextPart(
    val text: String,
    val thoughtSignature: String? = null,
    val isThought: Boolean = false,
)

/**
 * Text response of the model (without tool calls and without image inline data). Source of truth is
 * `textParts` (as it came from the API, no concatenation). The fields [text] and [thoughtSignature] are
 * derived conveniences for logs / JSON deserialization / further writing to historyPrompt;
 * they are assembled on the fly from `textParts` and do not go into serialization.
 */
@Serializable
data class TextPromptExecutionResult(
    val statistics: PromptExecutionStatistics,
    val textParts: List<ExecutionTextPart> = emptyList()
) : PromptExecutionResult {
    /** Full text of the "real" model response — without chain-of-thought pieces. */
    val text: String get() = textParts.asSequence().filterNot { it.isThought }.joinToString("") { it.text }

    /**
     * First encountered non-null thought signature. Gemini usually attaches the signature
     * to one of the thought blocks; one is enough for a follow-up request. If the
     * response has no thoughts — `null`.
     */
    val thoughtSignature: String? get() = textParts.firstNotNullOfOrNull { it.thoughtSignature }
}

/**
 * Result of an image-generating prompt. Sealed: a [Bytes] result still carries raw binary
 * payload in memory, a [File] result has been persisted by a `BinaryDataStorage` and only
 * references the file by [File.uri] — consumers should prefer reading from the URI in that
 * case (e.g. send it as `MessagePart.File` to the next round-trip instead of re-uploading
 * the bytes). Aggregator picks the variant based on what events were observed during the
 * stream.
 */
sealed interface ImagePromptExecutionResult : PromptExecutionResult {
    val mimeType: String
    val statistics: PromptExecutionStatistics
    val id: UUID

    /** Inline payload as it came from the provider (no storage in the chain). */
    @Serializable
    data class Bytes(
        @Transient
        val imageBytes: ByteArray = byteArrayOf(),
        override val mimeType: String,
        override val statistics: PromptExecutionStatistics,
        @Serializable(with = UUIDSerializer::class)
        override val id: UUID = UUID.randomUUID(),
    ) : ImagePromptExecutionResult

    /** Storage-persisted reference: the bytes already live behind [uri]. */
    @Serializable
    data class File(
        val uri: String,
        override val mimeType: String,
        override val statistics: PromptExecutionStatistics,
        @Serializable(with = UUIDSerializer::class)
        override val id: UUID = UUID.randomUUID(),
    ) : ImagePromptExecutionResult
}

@Serializable(with = ToolCallSerializer::class)
data class ToolCall(
    val name: String,
    val args: Map<String, JsonElement>,
    val thoughtSignature: String? = null,
    val callId: CallId,
)

@Serializable
data class ToolCallPromptExecutionResult(
    val toolCalls: List<ToolCall>,
    val statistics: PromptExecutionStatistics,
    val textParts: List<ExecutionTextPart> = emptyList()
) : PromptExecutionResult

/**
 * Result of an embedding prompt: the dense vector plus execution statistics. [embedding]
 * length equals the configured output dimensionality.
 */
@Serializable
data class EmbeddingPromptExecutionResult(
    val embedding: FloatArray,
    val statistics: PromptExecutionStatistics,
) : PromptExecutionResult {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is EmbeddingPromptExecutionResult) return false
        return embedding.contentEquals(other.embedding) && statistics == other.statistics
    }

    override fun hashCode(): Int = 31 * embedding.contentHashCode() + statistics.hashCode()
}

enum class ImageResolution(val pixels: Int) {
    SD_512(512), HD_1024(1024), XHD_2048(2048), UHD_4096(4096),
}

/**
 * Aspect ratio enum aligned with Imagen "W:H" strings. `widthRatio`/`heightRatio` carry the
 * numeric ratio directly — same numbers as appear in the API string ([widthRatio]:[heightRatio]).
 * The `PORTRAIT_*` / `LANDSCAPE_*` prefix is a label only; the numbers tell you which
 * dimension is which.
 */
enum class ImageAspectRatio(val widthRatio: Int, val heightRatio: Int) {
    SQUARE_1_1(1, 1),
    PORTRAIT_9_16(9, 16), LANDSCAPE_16_9(16, 9),
    PORTRAIT_4_3(4, 3), LANDSCAPE_3_4(3, 4),
    PORTRAIT_16_9(16, 9),    // numerically same as LANDSCAPE_16_9; kept as separate provider tag.
    LANDSCAPE_9_16(9, 16),   // numerically same as PORTRAIT_9_16; kept as separate provider tag.
    PORTRAIT_8_1(8, 1), LANDSCAPE_1_8(1, 8),
    PORTRAIT_3_2(3, 2), LANDSCAPE_2_3(2, 3),
    PORTRAIT_4_5(4, 5), LANDSCAPE_5_4(5, 4),
    PORTRAIT_21_9(21, 9),
}

data class ImageSize(val resolution: ImageResolution, val aspectRatio: ImageAspectRatio) {
    /**
     * Pixel width of the rendered image. The longer side of the bounding box is set to
     * [ImageResolution.pixels]; the shorter side is scaled by the aspect ratio. So
     * [ImageResolution.HD_1024] + 16:9 → 1024×576, + 9:16 → 576×1024, + 1:1 → 1024×1024.
     *
     * Caveat: providers can clamp or quantize these to their own grid (e.g. Imagen 3 returns
     * 1408×768 instead of mathematically clean 1024×576 for 16:9). Treat the result as
     * "what we asked for", not "what the bytes are guaranteed to be". For tactical maps
     * (SQUARE_1_1) the equality holds exactly.
     */
    val pixelWidth: Int get() {
        val longest = maxOf(aspectRatio.widthRatio, aspectRatio.heightRatio)
        return resolution.pixels * aspectRatio.widthRatio / longest
    }

    val pixelHeight: Int get() {
        val longest = maxOf(aspectRatio.widthRatio, aspectRatio.heightRatio)
        return resolution.pixels * aspectRatio.heightRatio / longest
    }

    companion object {
        val DEFAULT = ImageSize(ImageResolution.HD_1024, ImageAspectRatio.SQUARE_1_1)
    }
}

/**
 * Contract of the LLM executor. Returns a cold `Flow<PromptExecutionEvent>` — the consumer
 * subscribes via `.collect` and handles events (`TextChunk`, `ThoughtChunk`,
 * `ToolCallReceived`, `InlineData`, `StatisticsUpdate`, `Completed` / `Failed`) as
 * they arrive. No final snapshot — whoever needs the full text / statistics accumulates
 * it in their own listener.
 *
 * The method is **not suspend** — `flow {}` is cold by construction, suspend logic lives inside collect.
 *
 * Error contract: an implementation maps ANY failure of the provider call (transport, HTTP,
 * parsing, timeouts) into [PromptExecutionException] (or a subclass) before it leaves the flow —
 * callers catch that type only, and fill `partialStats` from the events seen so far.
 * [kotlinx.coroutines.CancellationException] is the sole exception to the rule and is rethrown as is.
 */
fun interface PromptExecutor {
    fun execute(prompt: Prompt): Flow<PromptExecutionEvent>

    /**
     * Whether this executor serves [llm] (typically by provider). [RoutingPromptExecutor] routes on
     * it to pick the provider's executor. Default `true`: provider-agnostic decorators (which wrap a
     * real executor and sit above the router) and test stubs need not override.
     */
    fun supports(llm: Llm): Boolean = true
}

/**
 * Decorator that forces [PromptSettings.includeThoughts] on all prompts passing through the
 * executor. Used at the pipeline level: the pipeline wraps the shared executor and itself
 * decides whether chain-of-thought should be returned (depends on the `saveThinkingInResponse` setting).
 */
fun PromptExecutor.withIncludeThoughts(value: Boolean): PromptExecutor = PromptExecutor { prompt ->
    execute(prompt.withIncludeThoughts(value))
}

fun Prompt.withIncludeThoughts(value: Boolean): Prompt {
    val newSettings: PromptSettings? = when (val s = settings) {
        is ToolsPromptSettings -> s.copy(includeThoughts = value)
        is TypedPromptSettings<*> -> s.copy(includeThoughts = value)
        is ImagePromptSettings -> s.copy(includeThoughts = value)
        null -> ToolsPromptSettings(emptyList(), includeThoughts = value)
        else -> s
    }
    return Prompt(llm, newSettings, content)
}

/**
 * Executor decorator that pins [PromptSettings.thinkingLevel] on every prompt that flows
 * through it. Used at pipeline level to attach a per-role reasoning hint (e.g. planner →
 * MEDIUM) without touching the agent builders themselves. `null` is a no-op (leaves the
 * existing settings untouched), so it's safe to wire the decorator unconditionally.
 */
fun PromptExecutor.withThinkingLevel(level: ThinkingLevel?): PromptExecutor =
    if (level == null) this
    else PromptExecutor { prompt -> execute(prompt.withThinkingLevel(level)) }

fun Prompt.withThinkingLevel(level: ThinkingLevel): Prompt {
    val newSettings: PromptSettings = when (val s = settings) {
        is ToolsPromptSettings -> s.copy(thinkingLevel = level)
        is TypedPromptSettings<*> -> s.copy(thinkingLevel = level)
        is ImagePromptSettings -> s.copy(thinkingLevel = level)
        null -> ToolsPromptSettings(emptyList(), thinkingLevel = level)
        else -> s
    }
    return Prompt(llm, newSettings, content)
}

/**
 * Executor decorator that pins [PromptSettings.maxOutputTokens] on every prompt that flows
 * through it. Wire at pipeline level to bound a role's per-RT output (cap covers content +
 * thinking tokens combined — pick value comfortably above the model's typical thinking
 * budget). `null` is a no-op so the decorator is safe to wire unconditionally.
 *
 * Concrete purpose: kill runaway-loop blast radius. A misbehaving model that spams
 * identical tool-calls hits MAX_TOKENS → [MaxTokensException] → recovery, instead of
 * silently aggregating thousands of calls into pipeline state.
 */
fun PromptExecutor.withMaxOutputTokens(value: Int?): PromptExecutor =
    if (value == null) this
    else PromptExecutor { prompt -> execute(prompt.withMaxOutputTokens(value)) }

fun Prompt.withMaxOutputTokens(value: Int): Prompt {
    val newSettings: PromptSettings = when (val s = settings) {
        is ToolsPromptSettings -> s.copy(maxOutputTokens = value)
        is TypedPromptSettings<*> -> s.copy(maxOutputTokens = value)
        is ImagePromptSettings -> s.copy(maxOutputTokens = value)
        null -> ToolsPromptSettings(emptyList(), maxOutputTokens = value)
        else -> s
    }
    return Prompt(llm, newSettings, content)
}

/**
 * Embeds [size] into prompt as `ImagePromptSettings` — a signal to the executor that the prompt goes
 * to an Imagen model. If the prompt already has ImagePromptSettings, only the size is updated.
 */
fun Prompt.withImageSize(size: ImageSize): Prompt {
    val current = settings
    // thinkingLevel is always run through clampThinkingLevel against the CATALOG fact about the model:
    // some image models reject some tiers with a 400 (live probes 2026-07-15:
    // `gemini-3.1-flash-lite-image` on LOW → "Thinking level LOW is not supported"; MINIMAL and
    // HIGH are both supported). The single point through which ALL image prompts pass —
    // we clamp here, not in each caller. `null` (provider default) is left alone — we do not
    // invent a level.
    val newSettings = if (current is ImagePromptSettings) {
        current.copy(
            size = size,
            thinkingLevel = current.thinkingLevel?.let { llm.clampThinkingLevel(it) },
        )
    } else {
        ImagePromptSettings(
            size = size,
            // Image-model thoughts are kept (they used to be dropped when assembling the response). Thinking is NOT
            // disabled — we turn the level (thinkingLevel, NOT count): LOW by default, the caller can
            // override it via its own settings.
            includeThoughts = true,
            thinkingLevel = llm.clampThinkingLevel(current?.thinkingLevel ?: ThinkingLevel.LOW),
        )
    }
    return Prompt(llm, newSettings, content)
}

fun PromptExecutionResult.statistics(): PromptExecutionStatistics = when (this) {
    is TextPromptExecutionResult -> statistics
    is ImagePromptExecutionResult -> statistics
    is ToolCallPromptExecutionResult -> statistics
    is EmbeddingPromptExecutionResult -> statistics
}

fun PromptExecutionStatistics.mergeWith(other: PromptExecutionStatistics): PromptExecutionStatistics {
    return copy(
        attempts = attempts + other.attempts,
        tokens = PromptExecutionTokenStatistics(
            totalTokens = tokens.totalTokens + other.tokens.totalTokens,
            input = tokens.input + other.tokens.input,
            output = tokens.output + other.tokens.output
        ),
        success = success && other.success,
        errorMessage = other.errorMessage ?: errorMessage,
        modelName = other.modelName,
        // Per-iteration trace: keep the latest exchange. The aggregated record is meant
        // for token/duration accounting; the most recent request/response is the most
        // useful when debugging the final state.
        requestMessages = other.requestMessages.ifEmpty { requestMessages },
        responseMessage = other.responseMessage ?: responseMessage,
    )
}

fun String.cleanupJson(): String {
    return this.trim()
        .removePrefix("```json")
        .removePrefix("```")
        .removeSuffix("```")
        .trim()
}

fun interface PromptExecutionResultSerializer<T> {
    suspend fun serialize(result: PromptExecutionResult): T
}

class JsonPromptSerializer<T : Any>(private val serializer: KSerializer<T>) : PromptExecutionResultSerializer<T> {
    override suspend fun serialize(result: PromptExecutionResult): T {
        val text = when (result) {
            is TextPromptExecutionResult -> result.text
            else -> throw PromptExecutorException("Expected TextPromptExecutionResult, got ${result::class.simpleName}")
        }
        return defaultJson.decodeFromString(serializer, text.cleanupJson())
    }

    companion object {
        private val defaultJson = Json {
            ignoreUnknownKeys = true
            isLenient = true
            coerceInputValues = true
        }
    }
}

fun interface TextPromptSerializer<T> : PromptExecutionResultSerializer<T> {
    override suspend fun serialize(result: PromptExecutionResult): T {
        val text = when (result) {
            is TextPromptExecutionResult -> result.text
            else -> throw PromptExecutorException("Expected TextPromptExecutionResult, got ${result::class.simpleName}")
        }
        return serialize(text)
    }

    suspend fun serialize(text: String): T

    companion object {
        private val defaultJson = Json {
            ignoreUnknownKeys = true
            isLenient = true
            coerceInputValues = true
        }

        fun <T : Any> Json(serializer: KSerializer<T>): TextPromptSerializer<T> = object : TextPromptSerializer<T> {
            override suspend fun serialize(text: String): T =
                defaultJson.decodeFromString(serializer, text.cleanupJson())
        }
    }
}

fun JsonElement.toAny(): Any? {
    return when (this) {
        is JsonNull -> null
        is JsonPrimitive -> {
            if (isString) content
            else {
                content.toBooleanStrictOrNull() ?:
                content.toLongOrNull() ?:
                content.toDoubleOrNull() ?:
                content
            }
        }
        is JsonArray -> map { it.toAny() }
        is JsonObject -> mapValues { it.value.toAny() }
    }
}

fun Any?.toJsonElement(): JsonElement {
    if (this == null) return JsonNull
    if (this is JsonElement) return this

    return when (this) {
        // A `Unit`-returning tool (setters/mutators — updateHp, setSummary, ...) has nothing
        // meaningful to report back. Falling through to the `else` branch would stringify it as
        // "kotlin.Unit", which reads to the model as a garbled/failed result rather than a
        // confirmation — models (notably GPT-5.x, one tool call per round-trip) then redo the
        // already-applied mutation "just in case" (seen as updateHp/setSummary/updateMemory
        // fired 4-5x with identical args in a single turn). Render it as an explicit success ack.
        is Unit -> JsonPrimitive("OK")
        is Boolean -> JsonPrimitive(this)
        is Number -> JsonPrimitive(this)
        is String -> JsonPrimitive(this)
        is Iterable<*> -> JsonArray(this.map { it.toJsonElement() })
        is Map<*, *> -> JsonObject(this.entries.associate { it.key.toString() to it.value.toJsonElement() })
        else -> JsonPrimitive(this.toString())
    }
}
