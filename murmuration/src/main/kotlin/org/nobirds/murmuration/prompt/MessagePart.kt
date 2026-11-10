package org.nobirds.murmuration.prompt

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient
import kotlinx.serialization.json.JsonElement

/**
 * Provider-neutral representation of a single part of an LLM message — either a side of
 * the conversation (request) or the model's output (response). Used in
 * [PromptExecutionStatistics] for telemetry/audit so the application (and admin tooling) can
 * inspect what was sent and what came back without parsing provider-specific JSON.
 *
 * Binary payloads are kept on [InlineData.data] as a `@Transient` field — they live in
 * memory at runtime but are stripped from the JSON serialisation that lands in the DB
 * (only `mimeType` / `sizeBytes` survive). Persisting binaries is a separate concern.
 */
@Serializable
sealed interface MessagePart {

    @Serializable
    @SerialName("text")
    data class Text(
        val text: String,
    ) : MessagePart

    /**
     * Chain-of-thought content from a reasoning model — kept structurally separate from
     * [Text] so consumers never have to look at a flag to know which one they're holding.
     * The optional [signature] is the thinking-trace identifier providers expect back on the
     * next round-trip when re-using accumulated thoughts (Gemini "thoughtSignature").
     */
    @Serializable
    @SerialName("thought")
    data class Thought(
        val text: String,
        val signature: String? = null,
    ) : MessagePart

    @Serializable(with = MessageToolCallSerializer::class)
    data class ToolCall(
        val name: String,
        val args: Map<String, JsonElement>,
        val thoughtSignature: String? = null,
        val callId: CallId,
    ) : MessagePart

    @Serializable(with = MessageToolResultSerializer::class)
    data class ToolResult(
        val name: String,
        val result: JsonElement,
        val thoughtSignature: String? = null,
        val callId: CallId,
    ) : MessagePart

    @Serializable
    @SerialName("file")
    data class File(
        val uri: String,
        val mimeType: String? = null,
        val name: String? = null,
    ) : MessagePart

    @Serializable
    @SerialName("inline_data")
    data class InlineData(
        val mimeType: String,
        val sizeBytes: Long,
        val name: String? = null,
        @Transient
        val data: ByteArray? = null,
    ) : MessagePart {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is InlineData) return false
            return mimeType == other.mimeType && sizeBytes == other.sizeBytes && name == other.name
        }

        override fun hashCode(): Int {
            var result = mimeType.hashCode()
            result = 31 * result + sizeBytes.hashCode()
            result = 31 * result + (name?.hashCode() ?: 0)
            return result
        }
    }

    /**
     * A dense embedding vector produced by an embedding model — a response-only part (never
     * appears in a request / history). [dimension] is serialised for telemetry; the actual
     * [values] are kept `@Transient` (like [InlineData.data]) so the vector payload doesn't
     * bloat the persisted statistics row. The aggregator reads [values] in-memory to build
     * the embedding result.
     */
    @Serializable
    @SerialName("embedding")
    data class Embedding(
        val dimension: Int,
        @Transient
        val values: FloatArray = floatArrayOf(),
    ) : MessagePart {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Embedding) return false
            return dimension == other.dimension
        }

        override fun hashCode(): Int = dimension
    }
}

/**
 * A single message in an LLM conversation: a role and the parts that compose it.
 * The request side of [PromptExecutionStatistics] is a list of these (system + user/model
 * history); the response side is a single [PromptMessage] with role = `MODEL`.
 */
@Serializable
data class PromptMessage(
    val role: PromptContentType,
    val parts: List<MessagePart>,
)

/**
 * Converts a framework-level [PromptContent] (system / user / model / tool message built
 * via [PromptBuilder]) into the provider-neutral [PromptMessage] form used by
 * [PromptExecutionStatistics.requestMessages].
 *
 * Binary [DataPromptPart] is dropped — for post-hoc prompt analysis the bytes are useless,
 * and persisting them inline would blow up `raw_request_prompt`. If you need the bytes, go
 * to [BinaryDataStorage].
 */
fun PromptContent.toMessage(): PromptMessage = PromptMessage(
    role = type,
    parts = parts.mapNotNull { it.toMessagePart() }
)

private fun PromptPart.toMessagePart(): MessagePart? = when (this) {
    is TextPromptPart -> MessagePart.Text(text = text)
    is ToolCallPromptPart -> MessagePart.ToolCall(
        name = call.name,
        args = call.args,
        thoughtSignature = call.thoughtSignature,
        callId = call.callId,
    )
    is ToolResultPromptPart -> MessagePart.ToolResult(
        name = name,
        result = result,
        thoughtSignature = thoughtSignature,
        callId = callId,
    )
    is FileDataPromptPart -> MessagePart.File(
        uri = uri,
        mimeType = mimeType,
        name = name,
    )
    is DataPromptPart -> null
}
