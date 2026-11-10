package org.nobirds.murmuration.gemini

import com.google.genai.types.Blob
import com.google.genai.types.Content
import com.google.genai.types.FileData
import com.google.genai.types.FunctionCall
import com.google.genai.types.FunctionResponse
import com.google.genai.types.Part
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import org.nobirds.murmuration.gemini.internal.getMimeTypeFromUrl
import org.nobirds.murmuration.prompt.DataPromptPart
import org.nobirds.murmuration.prompt.FileDataPromptPart
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptPart
import org.nobirds.murmuration.prompt.callsLast
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.ToolCallPromptPart
import org.nobirds.murmuration.prompt.ToolCall
import org.nobirds.murmuration.prompt.ToolResultPromptPart
import org.nobirds.murmuration.prompt.toAny
import java.util.Base64

/**
 * Converts prompt history to genai content. The only place that turns a [ToolCallPromptPart] into a
 * wire `FunctionCall`: tool-call arguments are renamed strictly by the tool's [WireKeyMap] from the same
 * [GeminiToolDeclarations.build] that produced the declarations, so history and declarations always
 * share one layout. A tool without a map (not mounted in this request) keeps canonical argument keys.
 */
class GeminiHistoryConverter(private val wireKeyMaps: Map<String, WireKeyMap>) {

    fun toGenaiContent(content: PromptContent, roleName: String? = null): Content {
        val parts = content.parts.map { toGenai(it) }
        val builder = Content.builder().parts(parts)
        roleName?.let { builder.role(it) }
        return builder.build()
    }

    /**
     * Builds the Gemini `systemInstruction` (a role-less [Content]) from SYSTEM prompt parts; null when there are none.
     * The single place used both by the request path and by cache baking.
     */
    fun toSystemInstruction(parts: List<PromptPart>): Content? =
        if (parts.isEmpty()) null else Content.builder().parts(parts.map { toGenai(it) }).build()

    fun toGenaiContents(contents: List<PromptContent>): List<Content> {
        return contents.mapNotNull { content ->
            if (content.parts.isEmpty()) return@mapNotNull null
            val role = when (content.type) {
                PromptContentType.USER -> "user"
                PromptContentType.MODEL -> "model"
                // Vertex's live generateContent path only accepts role user/model ("Please use a
                // valid role: user, model" 400 otherwise) — "function" is not a valid Content role
                // there, unlike some other Gemini surfaces. Tool exchange history is sent as user
                // content, the same as the cache-baking path, which goes through this function too.
                PromptContentType.TOOL -> "user"
                else -> "user"
            }
            Content.builder()
                .role(role)
                .parts(content.parts.laidOut(content.type).map { toGenai(it) })
                .build()
        }
    }

    /** The agent keeps a turn in arrival order; Gemini's wire wants a model turn with calls last. */
    private fun List<PromptPart>.laidOut(type: PromptContentType): List<PromptPart> =
        if (type == PromptContentType.MODEL) callsLast() else this

    fun toGenai(part: PromptPart): Part {
        return when (part) {
            is TextPromptPart -> {
                val builder = Part.builder().text(part.text)
                if (part.thoughtSignature != null) {
                    builder.thought(true)
                    builder.thoughtSignature(Base64.getDecoder().decode(part.thoughtSignature))
                }
                builder.build()
            }

            is DataPromptPart -> {
                Part.builder().inlineData(
                    Blob.builder()
                        .mimeType(part.mimeType)
                        .data(part.data) // data takes a byte[]
                        .build()
                ).build()
            }

            is FileDataPromptPart -> {
                Part.builder().fileData(
                    FileData.builder()
                        .apply {
                            val effectiveMimeType = if (part.mimeType == "application/octet-stream") {
                                getMimeTypeFromUrl(part.uri)
                            } else {
                                part.mimeType
                            }
                            effectiveMimeType?.let { mimeType(it) }
                        }
                        .fileUri(part.uri)
                        .build()
                ).build()
            }

            is ToolCallPromptPart -> {
                val builder = Part.builder().functionCall(
                    FunctionCall.builder()
                        .name(part.call.name)
                        .args(wireArgs(part.call).mapValues { it.value.toAny() })
                        .build()
                )
                part.call.thoughtSignature?.let { builder.thoughtSignature(Base64.getDecoder().decode(it)) }
                builder.build()
            }

            is ToolResultPromptPart -> {
                val builder = Part.builder().functionResponse(
                    FunctionResponse.builder()
                        .name(part.name)
                        .response(
                            if (part.result is JsonObject) part.result.toAny() as Map<String, Any?>
                            else mapOf("result" to part.result.toAny())
                        )
                        .build()
                )
                part.thoughtSignature?.let { builder.thoughtSignature(Base64.getDecoder().decode(it)) }
                builder.build()
            }
        }
    }

    private fun wireArgs(call: ToolCall): Map<String, JsonElement> =
        wireKeyMaps[call.name]?.toWire(call.args) ?: call.args

    companion object {
        /** Converter for requests without tools (no wire key maps): argument keys stay canonical. */
        val EMPTY = GeminiHistoryConverter(emptyMap())
    }
}
