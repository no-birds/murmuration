package org.nobirds.murmuration.gemini

import com.google.genai.types.Candidate
import com.google.genai.types.GenerateContentResponse
import com.google.genai.types.Part
import org.nobirds.murmuration.prompt.InputTokenStatistics
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.OutputTokenStatistics
import org.nobirds.murmuration.prompt.PromptCachingProvider
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutionEvent.StreamResponded
import org.nobirds.murmuration.prompt.PromptMessage
import java.util.Base64
import kotlin.jvm.optionals.getOrNull

/**
 * Converts one Gemini `Part` into a single [PromptExecutionEvent.PartReceived] (or `null`
 * for Parts that don't make sense in the output stream — `functionResponse` is request-side,
 * `fileData` in response is not produced by Gemini today).
 *
 * Mapping (the wrapped [MessagePart] tells the consumer what's inside):
 * - text + thought=true → `MessagePart.Thought(text, signature)`
 * - text + thought=false → `MessagePart.Text(text)`
 * - functionCall → `MessagePart.ToolCall(...)`
 * - inlineData (non-empty bytes) → `MessagePart.InlineData(mimeType, sizeBytes, data)`
 *
 * `thoughtSignature` is base64-encoded — same format that round-trips back via
 * `TextPromptPart.thoughtSignature`.
 */
internal fun Part.toEventOrNull(): PromptExecutionEvent? {
    text().getOrNull()?.let { text ->
        val signature = thoughtSignature().getOrNull()?.let { Base64.getEncoder().encodeToString(it) }
        val part: MessagePart = if (thought().getOrNull() == true)
            MessagePart.Thought(text, signature)
        else MessagePart.Text(text)
        return PartReceived(part)
    }
    functionCall().getOrNull()?.let { call ->
        val signature = thoughtSignature().getOrNull()?.let { Base64.getEncoder().encodeToString(it) }
        return PartReceived(call.toToolCall(signature))
    }
    inlineData().getOrNull()?.let { blob ->
        val bytes = blob.data().getOrNull() ?: return null
        return PartReceived(
            MessagePart.InlineData(
                mimeType = blob.mimeType().getOrNull() ?: "application/octet-stream",
                sizeBytes = bytes.size.toLong(),
                data = bytes,
            )
        )
    }
    // Gemini today doesn't return `fileData` in response (it's a request-side part for
    // referencing files in storage), but future providers (Claude, others) may. Emit the
    // file reference straight through — downstream agents will see an `ImageAgentResponse.File`
    // with the URI ready to be passed via `fileUri(...)` to the next round-trip.
    fileData().getOrNull()?.let { file ->
        val uri = file.fileUri().getOrNull() ?: return null
        return PartReceived(
            MessagePart.File(
                uri = uri,
                mimeType = file.mimeType().getOrNull(),
            )
        )
    }
    return null
}

/** All Parts from all candidates of the response in original order. */
internal fun GenerateContentResponse.allParts(): List<Part> =
    candidates().getOrNull()?.flatMap { it.partsOrEmpty() }.orEmpty()

internal fun Candidate.partsOrEmpty(): List<Part> =
    content().getOrNull()?.parts()?.getOrNull() ?: emptyList()

/**
 * Builds the [PromptStarted] event from request-side data only. Providers emit it once per
 * round-trip BEFORE the request is sent, so a round-trip that dies before the first byte
 * (TTFB timeout, connection error) still carries the request and cache data out. Token counts
 * are not part of it: input usage arrives through [PromptExecutionEvent.UsageDelta]
 * from the response/chunks.
 */
internal fun startedEvent(
    requestMessages: List<PromptMessage>,
    cacheResult: PromptCachingProvider.CacheResult?,
    requestedModelName: String?,
): PromptStarted = PromptStarted(
    modelName = requestedModelName ?: "",
    requestedModelName = requestedModelName,
    requestMessages = requestMessages,
    cacheId = cacheResult?.id,
    cacheStats = cacheResult?.statistics,
)

/**
 * Builds the [StreamResponded] event from the first sign of a Gemini answer — the first chunk of
 * a stream or the whole direct response. [GenerateContentResponse.modelVersion] is the model the
 * provider actually served; falls back to the requested name when absent.
 */
internal fun GenerateContentResponse.respondedEvent(requestedModelName: String?): StreamResponded =
    StreamResponded(servedModelName = modelVersion().getOrNull() ?: requestedModelName ?: "")

/**
 * Reads the cumulative input-side usage from a Gemini response chunk. `total` is the FULL
 * input (i.e. prompt tokens **including** the cached part).
 *
 * Per Vertex AI doc (`GenerateContentResponseUsageMetadata`):
 * - `promptTokenCount`: "The total number of tokens in the prompt. When `cachedContent` is
 *   set, this **also includes** the number of tokens in the cached content."
 * - `cachedContentTokenCount`: "The number of tokens in the cached content that was used
 *   for this request."
 *
 * So `promptTokenCount` is ALREADY total (includes cached). `cachedContentTokenCount` is a
 * subset breakdown. We pass `promptTokenCount` straight through as `total`; the previous
 * implementation summed them (treating `promptTokenCount` as non-cached) and double-counted
 * cached portion by 100%, inflating `tokens.input.total` and our cost reports by ~74K tokens
 * for cached-heavy calls. Empirically verified 2026-05-17 against actual Vertex billing.
 *
 * This is a **cumulative** read — the streaming strategy keeps a running snapshot and emits
 * the delta (current − previous) per chunk, the same way it handles output usage.
 */
internal fun GenerateContentResponse.readInputUsage(): InputTokenStatistics {
    val usage = usageMetadata().orElse(null) ?: return InputTokenStatistics.EMPTY
    val total = usage.promptTokenCount().orElse(0)?.toLong() ?: 0L
    val cached = usage.cachedContentTokenCount().orElse(0)?.toLong() ?: 0L
    return InputTokenStatistics(total = total, cached = cached, cacheWriteShort = 0, cacheWriteLong = 0)
}

/**
 * Cumulative output usage from a Gemini response. Providers compute deltas by subtracting
 * the previous cumulative snapshot before emitting [PromptExecutionEvent.UsageDelta].
 */
internal fun GenerateContentResponse.toOutputUsage(): OutputTokenStatistics {
    val usage = usageMetadata().orElse(null) ?: return OutputTokenStatistics.EMPTY
    val completion = usage.candidatesTokenCount().orElse(0)?.toLong() ?: 0L
    val thinking = usage.thoughtsTokenCount().orElse(0)?.toLong() ?: 0L
    return OutputTokenStatistics(total = completion, thinking = thinking)
}

/** Reads provider's `finish_reason` token (e.g. `STOP`, `MAX_TOKENS`) from the response. */
internal fun GenerateContentResponse.finishReasonString(): String? =
    candidates().orElse(emptyList()).firstOrNull()?.finishReason()?.getOrNull()?.toString()
