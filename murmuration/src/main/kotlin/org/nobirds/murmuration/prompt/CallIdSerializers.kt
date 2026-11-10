package org.nobirds.murmuration.prompt

import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.JsonElement
import org.nobirds.murmuration.tooling.RecordReader
import org.nobirds.murmuration.tooling.ToolOutcome

/*
 * The carriers of a [CallId] have NO default for it in their constructors, so a forgotten id is a
 * compile error. Wire names equal the originals.
 *
 * - [MessagePart.ToolCall] / [MessagePart.ToolResult] (the stored trace, `llm_statistics` /
 *   `llm_caches`): the id is NOT written — the trace format does not change — and is always read
 *   back as [CallId.LEGACY].
 * - [ToolCall] / [ToolOutcome] (file dumps): the id is written; a record that lacks the field
 *   (written before call ids existed) is read through a surrogate whose `callId` defaults to
 *   [CallId.LEGACY] — the only place a missing id is tolerated, and only for reading/display.
 */

@Serializable
@SerialName("tool_call")
internal class MessageToolCallSurrogate(
    val name: String,
    val args: Map<String, JsonElement>,
    val thoughtSignature: String? = null,
)

internal object MessageToolCallSerializer : KSerializer<MessagePart.ToolCall> {
    override val descriptor: SerialDescriptor = MessageToolCallSurrogate.serializer().descriptor
    override fun serialize(encoder: Encoder, value: MessagePart.ToolCall) = encoder.encodeSerializableValue(
        MessageToolCallSurrogate.serializer(),
        MessageToolCallSurrogate(value.name, value.args, value.thoughtSignature),
    )
    override fun deserialize(decoder: Decoder): MessagePart.ToolCall =
        decoder.decodeSerializableValue(MessageToolCallSurrogate.serializer())
            .let { MessagePart.ToolCall(it.name, it.args, it.thoughtSignature, CallId.LEGACY) }
}

@Serializable
@SerialName("tool_result")
internal class MessageToolResultSurrogate(
    val name: String,
    val result: JsonElement,
    val thoughtSignature: String? = null,
)

internal object MessageToolResultSerializer : KSerializer<MessagePart.ToolResult> {
    override val descriptor: SerialDescriptor = MessageToolResultSurrogate.serializer().descriptor
    override fun serialize(encoder: Encoder, value: MessagePart.ToolResult) = encoder.encodeSerializableValue(
        MessageToolResultSurrogate.serializer(),
        MessageToolResultSurrogate(value.name, value.result, value.thoughtSignature),
    )
    override fun deserialize(decoder: Decoder): MessagePart.ToolResult =
        decoder.decodeSerializableValue(MessageToolResultSurrogate.serializer())
            .let { MessagePart.ToolResult(it.name, it.result, it.thoughtSignature, CallId.LEGACY) }
}

@Serializable
@SerialName("org.nobirds.murmuration.prompt.ToolCall")
internal class ToolCallSurrogate(
    val name: String,
    val args: Map<String, JsonElement>,
    val thoughtSignature: String? = null,
    val callId: CallId = CallId.LEGACY,
)

internal object ToolCallSerializer : KSerializer<ToolCall> {
    override val descriptor: SerialDescriptor = ToolCallSurrogate.serializer().descriptor
    override fun serialize(encoder: Encoder, value: ToolCall) = encoder.encodeSerializableValue(
        ToolCallSurrogate.serializer(),
        ToolCallSurrogate(value.name, value.args, value.thoughtSignature, value.callId),
    )
    override fun deserialize(decoder: Decoder): ToolCall =
        decoder.decodeSerializableValue(ToolCallSurrogate.serializer())
            .let { ToolCall(it.name, it.args, it.thoughtSignature, it.callId) }
}

@Serializable
@SerialName("org.nobirds.murmuration.tooling.ToolOutcome")
internal class ToolOutcomeSurrogate(
    val name: String,
    val args: Map<String, JsonElement>,
    val rendered: JsonElement,
    val isError: Boolean,
    val ordinal: Int,
    val invoked: Boolean,
    val exit: ToolOutcome.Exit,
    val declaredReader: RecordReader?,
    val thoughtSignature: String? = null,
    val callId: CallId = CallId.LEGACY,
)

internal object ToolOutcomeSerializer : KSerializer<ToolOutcome> {
    override val descriptor: SerialDescriptor = ToolOutcomeSurrogate.serializer().descriptor
    override fun serialize(encoder: Encoder, value: ToolOutcome) = encoder.encodeSerializableValue(
        ToolOutcomeSurrogate.serializer(),
        ToolOutcomeSurrogate(
            value.name, value.args, value.rendered, value.isError, value.ordinal, value.invoked,
            value.exit, value.declaredReader, value.thoughtSignature, value.callId,
        ),
    )
    override fun deserialize(decoder: Decoder): ToolOutcome =
        decoder.decodeSerializableValue(ToolOutcomeSurrogate.serializer()).let {
            ToolOutcome(
                it.name, it.args, it.rendered, it.isError, it.ordinal, it.invoked,
                it.exit, it.declaredReader, it.thoughtSignature, it.callId,
            )
        }
}
