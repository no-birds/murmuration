package org.nobirds.murmuration.anthropic

import com.anthropic.core.http.StreamResponse
import com.anthropic.models.messages.RawContentBlockDeltaEvent
import com.anthropic.models.messages.RawContentBlockStartEvent
import com.anthropic.models.messages.RawMessageStreamEvent
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runInterruptible
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.jsonObject
import org.nobirds.murmuration.prompt.CallId
import org.nobirds.murmuration.prompt.MalformedResponseException
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.ParsePhase
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutionEvent.StreamResponded
import org.nobirds.murmuration.prompt.PromptMessage
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import kotlin.jvm.optionals.getOrNull
import kotlin.time.Duration
import kotlin.time.Duration.Companion.nanoseconds

/**
 * Translates the raw Messages API stream into core [PromptExecutionEvent]s, event by event. Pure state machine: no
 * network, no clock — the stream source and the timeouts live in [anthropicEventFlow].
 *
 * Emission rules (content blocks are keyed by `index`, so parallel `tool_use` blocks never mix):
 * - a `text` block goes out as ONE [MessagePart.Text] when the block closes (its `text_delta`s are concatenated: the unit of a
 *   history text part is the content block, not the stream delta — otherwise a long answer replays as dozens of micro-blocks);
 * - a `thinking` block goes out as ONE [MessagePart.Thought] when the block closes — its `signature_delta` arrives
 *   last, and a thought is only replayable with its signature (carrier: [AnthropicThoughtCarrier.encodeSignature]);
 * - a `redacted_thinking` block goes out as a [MessagePart.Thought] with empty text and the
 *   [AnthropicThoughtCarrier.encodeRedacted] carrier;
 * - a `tool_use` block goes out as [MessagePart.ToolCall] (`CallId = tool_use.id`) when the block closes, with the
 *   `input_json_delta` pieces concatenated and parsed;
 * - other block kinds (server tools) are not used by this provider and are ignored.
 *
 * `message_start` — always the first event of a Messages stream — is the provider's answer: [StreamResponded] with the served
 * model, followed by the input usage. [PromptStarted] is not the parser's: it is emitted by [anthropicEventFlow] before the
 * request is sent.
 *
 * Usage is mapped by [AnthropicUsageTracker]. [Completed] carries the raw `stop_reason`; judging it (refusal,
 * max_tokens, empty answer) belongs to the executor.
 */
internal class AnthropicStreamParser(private val requestedModelName: String) {
    private sealed interface Block {
        class Text(val text: StringBuilder) : Block
        class Thinking(val text: StringBuilder, var signature: String?) : Block
        class ToolUse(val id: String, val name: String, val json: StringBuilder) : Block
    }

    private val blocks = mutableMapOf<Long, Block>()
    private val usage = AnthropicUsageTracker()
    private var stopReason: String? = null

    /** True once `message_stop` was seen — a stream that ends without it is incomplete. */
    var finished: Boolean = false
        private set

    fun accept(event: RawMessageStreamEvent): List<PromptExecutionEvent> = when {
        event.isMessageStart() -> {
            val message = event.asMessageStart().message()
            listOf(
                StreamResponded(servedModelName = message.model().asString().ifEmpty { requestedModelName }),
                usage.onMessageStart(message.usage()),
            )
        }

        event.isContentBlockStart() -> onBlockStart(event.asContentBlockStart())
        event.isContentBlockDelta() -> onBlockDelta(event.asContentBlockDelta())
        event.isContentBlockStop() -> onBlockStop(event.asContentBlockStop().index())

        event.isMessageDelta() -> {
            val delta = event.asMessageDelta()
            delta.delta().stopReason().getOrNull()?.let { stopReason = it.asString() }
            listOf(usage.onMessageDelta(delta.usage()))
        }

        event.isMessageStop() -> {
            finished = true
            listOf(Completed(finishReason = stopReason))
        }

        else -> emptyList()
    }

    private fun onBlockStart(event: RawContentBlockStartEvent): List<PromptExecutionEvent> {
        val index = event.index()
        val block = event.contentBlock()
        return when {
            block.isText() -> {
                blocks[index] = Block.Text(StringBuilder(block.asText().text()))
                emptyList()
            }

            block.isThinking() -> {
                val thinking = block.asThinking()
                blocks[index] = Block.Thinking(StringBuilder(thinking.thinking()), thinking.signature().takeIf { it.isNotEmpty() })
                emptyList()
            }

            block.isRedactedThinking() -> listOf(
                PartReceived(MessagePart.Thought(text = "", signature = AnthropicThoughtCarrier.encodeRedacted(block.asRedactedThinking().data())))
            )

            block.isToolUse() -> {
                val toolUse = block.asToolUse()
                blocks[index] = Block.ToolUse(toolUse.id(), toolUse.name(), StringBuilder())
                emptyList()
            }

            else -> emptyList()
        }
    }

    private fun onBlockDelta(event: RawContentBlockDeltaEvent): List<PromptExecutionEvent> {
        val delta = event.delta()
        val block = blocks[event.index()]
        when {
            delta.isText() -> (block as? Block.Text)?.text?.append(delta.asText().text())
            delta.isThinking() -> (block as? Block.Thinking)?.text?.append(delta.asThinking().thinking())
            delta.isSignature() -> (block as? Block.Thinking)?.signature = delta.asSignature().signature()
            delta.isInputJson() -> (block as? Block.ToolUse)?.json?.append(delta.asInputJson().partialJson())
        }
        return emptyList()
    }

    private fun onBlockStop(index: Long): List<PromptExecutionEvent> = when (val block = blocks.remove(index)) {
        is Block.Text -> if (block.text.isEmpty()) emptyList() else listOf(PartReceived(MessagePart.Text(block.text.toString())))

        is Block.Thinking -> listOf(
            PartReceived(
                MessagePart.Thought(
                    text = block.text.toString(),
                    signature = block.signature?.let(AnthropicThoughtCarrier::encodeSignature),
                )
            )
        )

        is Block.ToolUse -> listOf(
            PartReceived(
                MessagePart.ToolCall(
                    name = block.name,
                    args = parseToolArgs(block.json.toString()),
                    callId = CallId(block.id),
                )
            )
        )

        null -> emptyList()
    }

    /** Empty arguments (a tool without parameters) are an empty map; unparseable ones are a retryable malformed response. */
    private fun parseToolArgs(json: String): Map<String, JsonElement> {
        val trimmed = json.trim()
        if (trimmed.isEmpty()) return emptyMap()
        return try {
            Json.parseToJsonElement(trimmed).jsonObject
        } catch (e: Exception) {
            throw MalformedResponseException(ParsePhase.FUNCTION_CALL, "Model returned unparseable tool-call arguments: ${e.message}", e)
        }
    }
}

/**
 * Streams one Messages API call as core events, [PromptStarted] first. The blocking SDK stream is read on `Dispatchers.IO`, one event at a time;
 * [idleTimeout] bounds the silence between events (`null` = no idle bound: a hidden Claude thought legally
 * silences the whole stream) and [overallTimeout] the whole round-trip (a stream that keeps trickling never trips the idle
 * one) — both end in a retryable [TransientException] (NETWORK_TIMEOUT). Cancelling the collector is never a timeout:
 * it surfaces as `CancellationException`, never as a mapped I/O error (see [interruptibleIo]).
 *
 * A stream with no events is EMPTY_STREAM, one that ends before `message_stop` is INCOMPLETE_RESPONSE (both transient).
 * SDK and mapping exceptions propagate as they are: translating them into the core exception types, emitting
 * `Failed` and judging `stop_reason` is the executor's job. The stream is always closed.
 */
internal fun anthropicEventFlow(
    openStream: () -> StreamResponse<RawMessageStreamEvent>,
    requestMessages: List<PromptMessage>,
    requestedModelName: String,
    idleTimeout: Duration?,
    overallTimeout: Duration,
): Flow<PromptExecutionEvent> = flow {
    val parser = AnthropicStreamParser(requestedModelName)
    val deadline = Deadline(System.nanoTime() + overallTimeout.inWholeNanoseconds, overallTimeout)
    var response: StreamResponse<RawMessageStreamEvent>? = null
    // Before the request leaves: a round-trip that dies before the first byte still carries the request out.
    emit(PromptStarted(modelName = requestedModelName, requestedModelName = requestedModelName, requestMessages = requestMessages))
    try {
        response = bounded(deadline, null) { openStream() }
        val iterator = response.stream().iterator()
        var eventCount = 0
        while (true) {
            val event = bounded(deadline, idleTimeout) { if (iterator.hasNext()) iterator.next() else null } ?: break
            eventCount++
            parser.accept(event).forEach { emit(it) }
        }
        if (eventCount == 0) {
            throw TransientException(TransientReason.EMPTY_STREAM, "Anthropic streaming returned 0 events before stream end", null)
        }
        if (!parser.finished) {
            throw TransientException(TransientReason.INCOMPLETE_RESPONSE, "Anthropic stream ended without message_stop", null)
        }
    } finally {
        runCatching { response?.close() }
    }
}

/**
 * Blocking [block] on `Dispatchers.IO`, interruptible by cancellation. `runInterruptible` turns only an
 * `InterruptedException` into a cancellation; okhttp reports the interrupt as `InterruptedIOException`, which the SDK wraps
 * into an `AnthropicIoException`. So a failure that arrives while the coroutine is already cancelled is the cancellation
 * (our own idle timeout or the caller's), not an I/O error: it is rethrown as the real cancellation.
 */
private suspend fun <T> interruptibleIo(block: () -> T): T = try {
    runInterruptible(Dispatchers.IO, block = block)
} catch (e: Exception) {
    currentCoroutineContext().ensureActive()
    throw e
}

/** The overall round-trip budget: absolute [atNanos] on the `System.nanoTime()` clock, [budget] only for the message. */
private class Deadline(val atNanos: Long, val budget: Duration) {
    fun remaining(): Duration = (atNanos - System.nanoTime()).nanoseconds
}

/**
 * One blocking read, bounded by the overall [deadline] AND by [idleTimeout] (`null` = no idle bound). The deadline cuts the read
 * itself (the coroutine is cancelled, the blocked thread interrupted), so a single hung read with the idle bound off cannot
 * outlive the budget until the SDK's own timeouts. Only our own expiry (`withTimeoutOrNull` lets foreign timeouts and
 * cancellations through) is a NETWORK_TIMEOUT: overall when the deadline has passed, idle otherwise.
 */
private suspend fun <T> bounded(deadline: Deadline, idleTimeout: Duration?, read: () -> T): T {
    val remaining = deadline.remaining()
    val limit = idleTimeout?.coerceAtMost(remaining) ?: remaining
    val result = if (limit.isPositive()) withTimeoutOrNull(limit) { Read(interruptibleIo(read)) } else null
    if (result != null) return result.value
    if (!deadline.remaining().isPositive()) {
        throw TransientException(
            TransientReason.NETWORK_TIMEOUT,
            "Anthropic streaming exceeded the overall round-trip budget of ${deadline.budget.inWholeSeconds}s",
            null,
        )
    }
    throw TransientException(TransientReason.NETWORK_TIMEOUT, "Anthropic streaming idle: no event within ${idleTimeout?.inWholeSeconds}s", null)
}

private class Read<T>(val value: T)
