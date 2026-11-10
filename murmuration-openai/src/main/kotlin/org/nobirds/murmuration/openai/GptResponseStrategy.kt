package org.nobirds.murmuration.openai

import com.openai.client.OpenAIClient
import com.openai.core.http.StreamResponse
import com.openai.models.responses.Response
import com.openai.models.responses.ResponseCreateParams
import com.openai.models.responses.ResponseReasoningItem
import com.openai.models.responses.ResponseStreamEvent
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.future.await
import kotlinx.coroutines.runInterruptible
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.jsonObject
import org.nobirds.murmuration.openai.internal.logger
import org.nobirds.murmuration.prompt.CallIds
import org.nobirds.murmuration.prompt.MalformedResponseException
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.ParsePhase
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Failed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutionEvent.StreamResponded
import org.nobirds.murmuration.prompt.PromptMessage
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.StreamTimingTracker
import org.nobirds.murmuration.prompt.finishReasonOrNull
import kotlin.jvm.optionals.getOrNull
import kotlin.time.Duration

/**
 * Strategy for obtaining a Responses API answer from Azure OpenAI: a single request
 * ([DirectGptStrategy]) or a streamed SSE response ([StreamingGptStrategy]). Emits
 * [PromptExecutionEvent] deltas as output items arrive; a terminal [Completed] / [Failed] closes
 * the flow. On any error a [Failed] is emitted before the exception propagates through the Flow.
 *
 * Mirror of `GeminiResponseStrategy`.
 */
interface GptResponseStrategy {
    fun execute(
        prompt: Prompt,
        params: ResponseCreateParams,
        requestMessages: List<PromptMessage>,
        requestedModelName: String,
    ): Flow<PromptExecutionEvent>
}

/**
 * Direct variant: one async `responses.create`. The whole response is decoded and its output
 * items emitted in a single pass — streaming semantics where every event arrives synchronously
 * after the single response returns.
 */
class DirectGptStrategy(
    private val client: OpenAIClient,
) : GptResponseStrategy {
    override fun execute(
        prompt: Prompt,
        params: ResponseCreateParams,
        requestMessages: List<PromptMessage>,
        requestedModelName: String,
    ): Flow<PromptExecutionEvent> = flow {
        // Before the request goes out: a request that fails without an answer keeps its request data.
        emit(
            PromptStarted(
                modelName = requestedModelName,
                requestedModelName = requestedModelName,
                requestMessages = requestMessages,
            )
        )
        try {
            val response = client.async().responses().create(params).await()
            emit(StreamResponded(servedModelName = response.modelName(requestedModelName)))
            // Output items arrive in order: reasoning first, then message / function calls.
            // Each reasoning item's signature rides on the FOLLOWING function-call part
            // (Gemini-style; see [ReasoningSignatureQueue]) — ToolCall parts round-trip their
            // `thoughtSignature` through the core agent loop verbatim AND visibly (telemetry
            // serialises it), unlike an empty Thought part.
            val reasoningQueue = ReasoningSignatureQueue()
            for (item in response.output()) {
                item.reasoning().getOrNull()?.let { reasoning ->
                    // Readable summary → telemetry-only Thought (marker signature, never
                    // replayed); the replayable encrypted signature rides the next tool call.
                    reasoning.toReadableThoughtOrNull()?.let { emit(PartReceived(it)) }
                    reasoningQueue.push(reasoning)
                }
                item.message().getOrNull()?.let { message ->
                    val text = message.content().mapNotNull { it.outputText().getOrNull()?.text() }.joinToString("")
                    if (text.isNotEmpty()) emit(PartReceived(MessagePart.Text(text)))
                }
                item.functionCall().getOrNull()?.let { call ->
                    emit(
                        PartReceived(
                            MessagePart.ToolCall(
                                name = call.name(),
                                args = parseToolArgs(call.arguments()),
                                thoughtSignature = reasoningQueue.pop(),
                                callId = CallIds.mint(),
                            )
                        )
                    )
                }
            }
            reasoningQueue.drainTo { emit(PartReceived(it)) }
            response.usage().getOrNull()?.let { emit(it.toUsageDelta()) }
            // A non-completed terminal status (incomplete/max_output_tokens, content_filter, failed)
            // must surface as an exception so the pipeline runs recovery instead of accepting a
            // truncated or policy-blocked response as success.
            response.mapOutcome(prompt)?.let { throw it }
            emit(Completed(response.finishToken()))
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            // `toPromptExecutionException` returns an already-typed exception unchanged, so an
            // outcome mapping thrown above passes through without remapping. A single Failed is
            // emitted for every error path (including tool-arg parse failures).
            val mapped = e.toPromptExecutionException(prompt)
            emit(Failed(mapped, finishReason = mapped.finishReasonOrNull()))
            throw mapped
        }
    }
}

/**
 * Streaming variant: `responses.createStreaming` returns a [StreamResponse] which we drain as a
 * Flow, emitting events as output items arrive.
 *
 * - Text deltas (`response.output_text.delta`) are emitted immediately as [MessagePart.Text].
 * - Readable reasoning summaries (`response.reasoning_summary_text.done`) are emitted as
 *   marker-signed [MessagePart.Thought] parts (telemetry only, see [toReadableThoughtOrNull]).
 * - Completed output items (`response.output_item.done`) carry the whole item: a reasoning
 *   item's encrypted signature is queued onto the FOLLOWING function_call part
 *   ([ReasoningSignatureQueue]); a function_call becomes a [MessagePart.ToolCall] with
 *   fully-assembled arguments. A message item is skipped — its text already streamed via
 *   deltas. Items arrive ordered by `output_index`, so emitting on each done preserves order
 *   (reasoning before its paired call/message).
 * - Usage + terminal status come from the final [Response] on the `completed` / `incomplete` /
 *   `failed` event.
 *
 * An idle timeout on each event turns silence longer than [idleTimeout] into a [TransientException]
 * (NETWORK_TIMEOUT) that the retry layer picks up. Per-prompt override via
 * [org.nobirds.murmuration.prompt.PromptSettings.streamIdleTimeout].
 *
 * On top of the per-event idle timeout, [overallTimeout] caps the **whole** round-trip. The idle
 * timeout alone is insufficient: every SSE event — including frequent reasoning deltas — resets it,
 * so a stuck request that keeps trickling reasoning events never trips idle and can hang far past
 * any sane budget (observed live: 12+ min). The overall cap is checked before each event pull and,
 * once exceeded, raises the same retryable [TransientException] (NETWORK_TIMEOUT). It is a hard
 * ceiling, not a per-event bound, so a legitimate long round-trip that steadily streams stays fine
 * as long as its total wall time stays under [overallTimeout].
 */
class StreamingGptStrategy(
    private val client: OpenAIClient,
    private val idleTimeout: Duration,
    private val overallTimeout: Duration,
) : GptResponseStrategy {

    private val logger = logger()

    override fun execute(
        prompt: Prompt,
        params: ResponseCreateParams,
        requestMessages: List<PromptMessage>,
        requestedModelName: String,
    ): Flow<PromptExecutionEvent> = flow {
        val effectiveIdleTimeout = prompt.settings?.streamIdleTimeout ?: idleTimeout

        var streamResponse: StreamResponse<ResponseStreamEvent>? = null
        val streamStartedAt = System.nanoTime()
        val overallDeadlineAt = streamStartedAt + overallTimeout.inWholeNanoseconds
        var chunkCount = 0
        var lastChunkAt = streamStartedAt
        // time-to-first-chunk / max inter-chunk gap, tracked unconditionally —
        // mirrors the Gemini side, see [StreamTimingTracker] KDoc.
        val timing = StreamTimingTracker(streamStartedAt)
        var totalTextParts = 0
        var totalThoughtParts = 0
        var totalToolCallParts = 0
        var finalResponse: Response? = null
        val reasoningQueue = ReasoningSignatureQueue()
        // Reasoning item ids whose readable summary already went out via
        // `response.reasoning_summary_text.done` — guards against double Thought emission
        // when the completed item also carries the summary.
        val summaryEmittedFor = mutableSetOf<String>()

        // Before the request goes out: a stream that dies before its first event keeps its request data.
        emit(
            PromptStarted(
                modelName = requestedModelName,
                requestedModelName = requestedModelName,
                requestMessages = requestMessages,
            )
        )
        try {
            streamResponse = client.responses().createStreaming(params)
            val iterator = streamResponse.stream().iterator()
            var respondedEmitted = false

            while (true) {
                // Hard overall cap on top of the per-event idle timeout: reasoning deltas keep
                // resetting idle, so only this total-wall-time check stops an endlessly trickling
                // stuck round-trip. NETWORK_TIMEOUT → retryable, same path as idle.
                if (System.nanoTime() >= overallDeadlineAt) {
                    throw TransientException(
                        TransientReason.NETWORK_TIMEOUT,
                        "Azure OpenAI streaming exceeded overall round-trip budget of " +
                            "${overallTimeout.inWholeSeconds}s (idle timeout kept resetting on reasoning deltas)",
                        null,
                    )
                }
                val event = iterator.nextOrNull(effectiveIdleTimeout) ?: break
                chunkCount++
                lastChunkAt = System.nanoTime()
                timing.recordChunk(lastChunkAt)

                // StreamResponded on the first event of the stream — normally `response.created`,
                // which also carries the served model name. A stream that opens with anything else
                // (defensive) still gets exactly one StreamResponded before its first part, named
                // after the requested model.
                if (!respondedEmitted) {
                    val served = event.created().getOrNull()?.response()?.modelName(requestedModelName) ?: requestedModelName
                    emit(StreamResponded(servedModelName = served))
                    respondedEmitted = true
                }

                event.outputTextDelta().getOrNull()?.let { delta ->
                    val text = delta.delta()
                    if (text.isNotEmpty()) {
                        totalTextParts++
                        emit(PartReceived(MessagePart.Text(text)))
                    }
                }
                // Readable reasoning summary: one Thought per completed summary part
                // (`response.reasoning_summary_text.done`), marker signature — telemetry only,
                // never replayed. Deltas are not re-emitted piecewise; the done event carries
                // the full text.
                event.reasoningSummaryTextDone().getOrNull()?.let { summaryDone ->
                    if (summaryDone.text().isNotEmpty()) {
                        totalThoughtParts++
                        summaryEmittedFor += summaryDone.itemId()
                        emit(
                            PartReceived(
                                MessagePart.Thought(
                                    text = summaryDone.text(),
                                    signature = encodeReasoningSignature(summaryDone.itemId(), null),
                                )
                            )
                        )
                    }
                }
                event.outputItemDone().getOrNull()?.let { done ->
                    val item = done.item()
                    item.reasoning().getOrNull()?.let { reasoning ->
                        totalThoughtParts++
                        // Fallback for servers that embed the summary only in the completed item
                        // (no separate summary_text events) — avoid double emission.
                        if (reasoning.id() !in summaryEmittedFor) {
                            reasoning.toReadableThoughtOrNull()?.let { emit(PartReceived(it)) }
                        }
                        reasoningQueue.push(reasoning)
                    }
                    item.functionCall().getOrNull()?.let { call ->
                        totalToolCallParts++
                        emit(
                            PartReceived(
                                MessagePart.ToolCall(
                                    name = call.name(),
                                    args = parseToolArgs(call.arguments()),
                                    thoughtSignature = reasoningQueue.pop(),
                                    callId = CallIds.mint(),
                                )
                            )
                        )
                    }
                    // A message item's text already streamed via output_text deltas — don't
                    // re-emit it here.
                }
                event.completed().getOrNull()?.let { finalResponse = it.response() }
                event.incomplete().getOrNull()?.let { finalResponse = it.response() }
                event.failed().getOrNull()?.let { finalResponse = it.response() }
                // A stream-level error event (rare) aborts the stream.
                event.error().getOrNull()?.let { err ->
                    throw TransientException(TransientReason.NETWORK_ERROR, err.message(), null)
                }
            }

            if (chunkCount == 0) throw TransientException(
                TransientReason.EMPTY_STREAM,
                "Azure OpenAI streaming returned 0 events before stream end",
                null,
            )

            val response = finalResponse ?: throw TransientException(
                TransientReason.INCOMPLETE_RESPONSE,
                "Azure OpenAI stream ended without a terminal response event",
                null,
            )
            // Reasoning items with no following function call (a terminal text answer) fall back
            // to the Thought channel so the signature still lands in telemetry/history.
            reasoningQueue.drainTo { emit(PartReceived(it)) }
            response.usage().getOrNull()?.let { emit(it.toUsageDelta()) }
            val token = response.finishToken()
            val diag = timing.snapshot(chunkCount, totalTextParts, totalThoughtParts, totalToolCallParts, token)
            response.mapOutcome(prompt)?.let { mapped ->
                mapped.streamDiagnostics = diag
                throw mapped
            }
            emit(Completed(token, streamDiagnostics = diag))
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            val lastToken = finalResponse?.finishToken()
            val diag = timing.snapshot(chunkCount, totalTextParts, totalThoughtParts, totalToolCallParts, lastToken)
            logger.warn(
                "Azure OpenAI stream failed [{}]: {} — chunks={}, total_elapsed={}ms, since_last_chunk={}ms, last_finish_reason={}, parts_text={}, parts_thought={}, parts_tool_call={}",
                e::class.simpleName, e.message, diag.chunkCount, diag.totalElapsedMs,
                diag.sinceLastChunkMs, diag.lastChunkFinishReason, diag.partsText, diag.partsThought, diag.partsToolCall,
            )
            val mapped = e.toPromptExecutionException(prompt)
            mapped.streamDiagnostics = diag
            emit(Failed(mapped, finishReason = mapped.finishReasonOrNull() ?: lastToken, streamDiagnostics = diag))
            throw mapped
        } finally {
            // Ignore close failures — OkHttp may throw if the socket was already closed by the
            // remote side; there is nothing left to recover here.
            runCatching { streamResponse?.close() }
        }
    }
}

/**
 * FIFO of reasoning-item signatures awaiting their paired following item. The Responses API
 * emits `[reasoning, function_call, reasoning, function_call, …]`; each reasoning item's
 * signature (id + `encrypted_content`, see [encodeReasoningSignature]) is attached to the NEXT
 * function-call part as [MessagePart.ToolCall.thoughtSignature] — the same channel Gemini's
 * `thoughtSignature` uses. The core agent loop already round-trips ToolCall signatures verbatim
 * into the history ([org.nobirds.murmuration.prompt.ToolCallPromptPart]), and [toResponseInput]
 * replays them as `ResponseInputItem.ofReasoning` immediately before the paired `function_call`
 * — the reasoning↔call pairing is atomic by construction.
 *
 * Signatures left over at the end of a turn (a terminal text answer with no following call)
 * drain as [MessagePart.Thought] fallback parts, so the signature still reaches telemetry and
 * the [org.nobirds.murmuration.prompt.TextPromptPart]-based replay branch of [toResponseInput].
 */
internal class ReasoningSignatureQueue {
    private val pending = ArrayDeque<ResponseReasoningItem>()

    fun push(item: ResponseReasoningItem) {
        pending.addLast(item)
    }

    /** Signature for the next function-call part, or `null` when no reasoning preceded it. */
    fun pop(): String? = pending.removeFirstOrNull()?.let { encodeReasoningSignature(it.id(), it.encryptedContent().getOrNull()) }

    /**
     * Emits every unpaired reasoning item as a signature-only [MessagePart.Thought] fallback
     * part (empty text — the readable summary, if any, already went out as its own
     * marker-signed Thought when the item arrived).
     */
    inline fun drainTo(emit: (MessagePart.Thought) -> Unit) {
        while (true) {
            val item = removeFirstOrNull() ?: return
            emit(
                MessagePart.Thought(
                    text = "",
                    signature = encodeReasoningSignature(item.id(), item.encryptedContent().getOrNull()),
                )
            )
        }
    }

    @PublishedApi
    internal fun removeFirstOrNull(): ResponseReasoningItem? = pending.removeFirstOrNull()
}

/**
 * The readable side of a reasoning item: its summary text as a [MessagePart.Thought] for
 * telemetry (the only human-readable window into GPT-5.x thoughts — raw CoT never leaves the
 * server), or `null` when the summary is empty. The signature is a **marker** (item id, no
 * encrypted content): [toResponseInput] recognises it as reasoning and neither replays it nor
 * leaks the text into the history as assistant prose. The replayable encrypted signature rides
 * the following ToolCall part instead (see [ReasoningSignatureQueue]).
 */
internal fun ResponseReasoningItem.toReadableThoughtOrNull(): MessagePart.Thought? {
    val text = summary().joinToString("\n") { it.text() }
    if (text.isEmpty()) return null
    return MessagePart.Thought(text = text, signature = encodeReasoningSignature(id(), null))
}

/** Best-effort model name off a [Response], falling back to the requested name. */
internal fun Response.modelName(fallback: String): String =
    runCatching { model().asString() }.getOrNull() ?: fallback

/**
 * Parses a tool-call `arguments` JSON string (as produced by the model) into the provider-neutral
 * arg map. Empty/blank arguments (a no-parameter function) map to an empty map; unparseable
 * arguments raise [MalformedResponseException] (FUNCTION_CALL), which the retry layer treats as
 * retryable (function-call sampling is non-deterministic).
 */
private fun parseToolArgs(argsJson: String): Map<String, JsonElement> {
    val trimmed = argsJson.trim()
    if (trimmed.isEmpty()) return emptyMap()
    return try {
        Json.parseToJsonElement(trimmed).jsonObject
    } catch (e: Exception) {
        throw MalformedResponseException(
            ParsePhase.FUNCTION_CALL,
            "Model returned unparseable tool-call arguments: ${e.message}",
            e,
        )
    }
}

/**
 * Pulls the next stream element on [Dispatchers.IO] with an idle timeout. Silence longer than
 * [idleTimeout] becomes a [TransientException] (NETWORK_TIMEOUT) the retry layer handles.
 * Returns `null` when the stream is exhausted. Copy of the gemini `nextOrNull` idiom.
 */
private suspend fun <T : Any> Iterator<T>.nextOrNull(idleTimeout: Duration): T? = try {
    withTimeout(idleTimeout) {
        runInterruptible(Dispatchers.IO) {
            if (hasNext()) next() else null
        }
    }
} catch (_: TimeoutCancellationException) {
    throw TransientException(
        TransientReason.NETWORK_TIMEOUT,
        "Azure OpenAI streaming idle: no event received within ${idleTimeout.inWholeSeconds}s (timeout/connection lost)",
        null,
    )
}
