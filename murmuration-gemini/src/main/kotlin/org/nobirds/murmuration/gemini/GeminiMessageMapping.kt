package org.nobirds.murmuration.gemini

import org.nobirds.murmuration.prompt.CallIds
import com.fasterxml.jackson.databind.JsonNode
import com.google.genai.types.Content
import com.google.genai.types.Part
import kotlinx.serialization.json.JsonObject
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptMessage
import org.nobirds.murmuration.prompt.toJsonElement
import java.util.Base64
import kotlin.jvm.optionals.getOrNull

/**
 * Convert a Gemini [Content] into a provider-neutral [PromptMessage] suitable for
 * statistics / telemetry storage. Binary blob bytes are kept on the runtime object
 * (`MessagePart.InlineData.data`) but are stripped from JSON serialisation by the
 * `@Transient` marker — only `mimeType` / `sizeBytes` survive into the DB dump.
 */
internal fun Content.toMessage(): PromptMessage {
    val role = role().getOrNull()?.toPromptContentType() ?: PromptContentType.USER
    val parts = parts().getOrNull()?.mapNotNull { it.toMessagePart() } ?: emptyList()
    return PromptMessage(role = role, parts = parts)
}

internal fun List<Content>.toMessages(): List<PromptMessage> = map { it.toMessage() }

private fun String.toPromptContentType(): PromptContentType = when (lowercase()) {
    "system" -> PromptContentType.SYSTEM
    "model" -> PromptContentType.MODEL
    // "tool" is the current role name; "function" is legacy — older dumps were written back
    // when toGenaiContents() (GeminiPromptExecutor.kt) still emitted role="function" for TOOL
    // content, before that got fixed to "user" to match what Vertex actually accepts.
    "tool", "function" -> PromptContentType.TOOL
    else -> PromptContentType.USER
}

internal fun Part.toMessagePart(): MessagePart? {
    val signature = thoughtSignature().getOrNull()?.let { Base64.getEncoder().encodeToString(it) }

    text().getOrNull()?.let { text ->
        return if (thought().getOrNull() == true)
            MessagePart.Thought(text = text, signature = signature)
        else MessagePart.Text(text = text)
    }

    functionCall().getOrNull()?.let { call -> return call.toToolCall(signature) }

    functionResponse().getOrNull()?.let { response ->
        val resultMap = response.response().getOrNull() ?: emptyMap()
        val resultJson = JsonObject(
            resultMap.mapValues { (_, v) -> if (v is JsonNode) v.toJsonElement() else v.toJsonElement() }
        )
        return MessagePart.ToolResult(
            name = response.name().getOrNull() ?: "",
            result = resultJson,
            thoughtSignature = signature,
            callId = CallIds.mint(),
        )
    }

    inlineData().getOrNull()?.let { blob ->
        val bytes = blob.data().getOrNull()
        return MessagePart.InlineData(
            mimeType = blob.mimeType().getOrNull() ?: "application/octet-stream",
            sizeBytes = bytes?.size?.toLong() ?: 0L,
            data = bytes,
        )
    }

    fileData().getOrNull()?.let { file ->
        return MessagePart.File(
            uri = file.fileUri().getOrNull() ?: "",
            mimeType = file.mimeType().getOrNull(),
        )
    }

    return null
}
