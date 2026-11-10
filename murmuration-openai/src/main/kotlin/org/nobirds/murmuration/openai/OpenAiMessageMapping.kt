package org.nobirds.murmuration.openai

import com.openai.models.responses.EasyInputMessage
import com.openai.models.responses.ResponseFunctionToolCall
import com.openai.models.responses.ResponseInputContent
import com.openai.models.responses.ResponseInputImage
import com.openai.models.responses.ResponseInputItem
import com.openai.models.responses.ResponseInputText
import com.openai.models.responses.ResponseReasoningItem
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import org.nobirds.murmuration.prompt.DataPromptPart
import org.nobirds.murmuration.prompt.FileDataPromptPart
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptMessage
import org.nobirds.murmuration.prompt.PromptPart
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.ToolCallPromptPart
import org.nobirds.murmuration.prompt.ToolResultPromptPart
import org.nobirds.murmuration.prompt.toMessage
import org.nobirds.murmuration.prompt.withCallsLast
import org.slf4j.LoggerFactory
import java.util.Base64

private val logger = LoggerFactory.getLogger("org.nobirds.murmuration.openai.OpenAiMessageMapping")

/**
 * A model reasoning item serialised onto a part's thought-signature so it can be replayed on the
 * next round-trip. [id] is the Responses item id (`rs_…`); [enc] is the opaque
 * `encrypted_content` blob that a `store = false` request returns and requires back verbatim.
 *
 * Encoding is a compact JSON string kept directly in the part's thought-signature slot —
 * primarily [org.nobirds.murmuration.prompt.ToolCall.thoughtSignature] on the function-call part
 * that followed the reasoning item (see `ReasoningSignatureQueue`), with
 * `MessagePart.Thought.signature` / [TextPromptPart.thoughtSignature] as the terminal-text
 * fallback. Same channel Gemini uses for its own thought signatures, so no core changes are
 * needed. See [encodeReasoningSignature] / [decodeReasoningSignature].
 */
@Serializable
internal data class ReasoningSignature(val id: String, val enc: String? = null)

private val signatureJson = Json { ignoreUnknownKeys = true }

internal fun encodeReasoningSignature(id: String, encryptedContent: String?): String =
    signatureJson.encodeToString(ReasoningSignature.serializer(), ReasoningSignature(id, encryptedContent))

/** Decodes a reasoning signature, or `null` if the string isn't one of ours (defensive). */
internal fun decodeReasoningSignature(signature: String?): ReasoningSignature? {
    if (signature.isNullOrEmpty()) return null
    return try {
        signatureJson.decodeFromString(ReasoningSignature.serializer(), signature)
    } catch (_: Exception) {
        null
    }
}

/**
 * Converts the framework's [PromptContent] history into an Azure OpenAI **Responses API** input
 * item list ([ResponseInputItem]).
 *
 * Role mapping:
 * - `SYSTEM` → a **developer** message ([EasyInputMessage] with role `DEVELOPER`). GPT-5.x
 *   reasoning models take instructions on the developer role; text parts join with a blank line.
 * - `USER` → a user message. Text-only content becomes an [EasyInputMessage] with a plain string;
 *   an image [DataPromptPart] (`image/…`) or an `http(s)` [FileDataPromptPart] promotes the whole
 *   message to a [ResponseInputItem.Message] carrying an `input_text` + `input_image` list.
 * - `MODEL` → reasoning items first, then assistant text, then function calls (see below).
 * - `TOOL` → one `function_call_output` per [ToolResultPromptPart], matched by position to the
 *   preceding model's function calls.
 *
 * **Reasoning replay & atomic pairing.** A reasoning item returned by the model rides the
 * **following function-call part** as [org.nobirds.murmuration.prompt.ToolCall.thoughtSignature]
 * (id + `encrypted_content`, see [ReasoningSignature] / `ReasoningSignatureQueue`) — the exact
 * channel Gemini's thought signatures use, carried verbatim by the core agent loop into
 * [ToolCallPromptPart] and, unlike an empty Thought part, visible in telemetry. Here, a
 * tool-call part whose signature decodes into a replayable [ReasoningSignature] is emitted as
 * [ResponseInputItem.ofReasoning] **immediately before** its own `function_call` — pairing is
 * atomic by construction: the reasoning item is derived from the very part it precedes, so it
 * can never appear without its required following item. (The Responses API rejects an orphan
 * reasoning item with a hard 400 `Item 'rs_…' was provided without its required following item`.)
 * A [TextPromptPart] carrying a reasoning signature (the terminal-text fallback, see
 * `ReasoningSignatureQueue.drainTo`) replays the same way, immediately before the turn's
 * remaining items. Whole tool-exchanges are also dropped atomically by per-section pruning
 * (`ToolCallingAgent` marks a turn's parts with one `replaceableSectionId`), so a replayed pair
 * never gets split by history maintenance either.
 *
 * **Synthetic call ids.** The framework's [org.nobirds.murmuration.prompt.ToolCall] /
 * [ToolResultPromptPart] carry no id, but the Responses API pairs a `function_call` to its
 * `function_call_output` via `call_id`. We mint ids `call_<n>` with a counter monotonic across
 * the whole history: each model function call gets the next id, and the tool results that follow
 * are matched **by position** (the `ToolCallingAgent` contract guarantees results arrive in call
 * order). A result with no matching call gets `call_orphan_<n>` and a warning.
 */
internal fun List<PromptContent>.toResponseInput(): List<ResponseInputItem> {
    val items = mutableListOf<ResponseInputItem>()
    var callCounter = 0
    var orphanCounter = 0
    var lastModelCallIds: List<String> = emptyList()

    for (content in withCallsLast()) {
        if (content.parts.isEmpty()) continue
        when (content.type) {
            PromptContentType.SYSTEM -> {
                val text = content.parts.filterIsInstance<TextPromptPart>().joinToString("\n\n") { it.text }
                if (text.isNotEmpty()) items += developerMessage(text)
            }

            PromptContentType.USER -> userMessage(content.parts)?.let { items += it }

            PromptContentType.MODEL -> {
                val callIds = mutableListOf<String>()
                for (part in content.parts) {
                    when (part) {
                        is TextPromptPart -> {
                            val reasoning = decodeReasoningSignature(part.thoughtSignature)
                            when {
                                // A fallback-channel reasoning item (terminal text turn): replay
                                // only if we still hold its encrypted content (a store=false
                                // reasoning item is useless without it).
                                reasoning != null -> reasoning.enc?.let { items += reasoningItem(reasoning.id, it) }
                                // Ordinary assistant text.
                                part.text.isNotEmpty() -> items += assistantMessage(part.text)
                            }
                        }

                        is ToolCallPromptPart -> {
                            // Reasoning replay: the signature riding on the call part becomes the
                            // reasoning item IMMEDIATELY BEFORE the function_call it belongs to —
                            // atomic pairing by construction (see the KDoc above).
                            decodeReasoningSignature(part.call.thoughtSignature)?.let { reasoning ->
                                reasoning.enc?.let { items += reasoningItem(reasoning.id, it) }
                            }
                            val id = "call_${callCounter++}"
                            callIds += id
                            items += ResponseInputItem.ofFunctionCall(
                                ResponseFunctionToolCall.builder()
                                    .callId(id)
                                    .name(part.call.name)
                                    .arguments(JsonObject(part.call.args).toString())
                                    .build()
                            )
                        }

                        else -> {}
                    }
                }
                lastModelCallIds = callIds
            }

            PromptContentType.TOOL -> {
                val results = content.parts.filterIsInstance<ToolResultPromptPart>()
                results.forEachIndexed { index, part ->
                    val id = lastModelCallIds.getOrNull(index) ?: run {
                        logger.warn(
                            "Tool result #{} for '{}' has no matching function_call id (only {} calls in the " +
                                "preceding model turn) — using a synthetic orphan id",
                            index, part.name, lastModelCallIds.size,
                        )
                        "call_orphan_${orphanCounter++}"
                    }
                    items += ResponseInputItem.ofFunctionCallOutput(
                        ResponseInputItem.FunctionCallOutput.builder()
                            .callId(id)
                            .output(part.result.toString())
                            .build()
                    )
                }
            }
        }
    }
    return items
}

/** A replayed reasoning input item: original item id + the opaque encrypted CoT blob. */
private fun reasoningItem(id: String, encryptedContent: String): ResponseInputItem =
    ResponseInputItem.ofReasoning(
        ResponseReasoningItem.builder()
            .id(id)
            .encryptedContent(encryptedContent)
            .summary(emptyList())
            .build()
    )

private fun developerMessage(text: String): ResponseInputItem =
    ResponseInputItem.ofEasyInputMessage(
        EasyInputMessage.builder().role(EasyInputMessage.Role.DEVELOPER).content(text).build()
    )

private fun assistantMessage(text: String): ResponseInputItem =
    ResponseInputItem.ofEasyInputMessage(
        EasyInputMessage.builder().role(EasyInputMessage.Role.ASSISTANT).content(text).build()
    )

/**
 * Builds a user message. Text-only content stays a plain-string [EasyInputMessage]; the presence
 * of any image promotes it to a [ResponseInputItem.Message] with an `input_text` + `input_image`
 * content list. Returns `null` if nothing representable is left.
 */
private fun userMessage(parts: List<PromptPart>): ResponseInputItem? {
    val hasImage = parts.any {
        (it is DataPromptPart && it.mimeType.startsWith("image/")) ||
            (it is FileDataPromptPart && (it.uri.startsWith("http://") || it.uri.startsWith("https://")))
    }
    if (!hasImage) {
        val text = parts.filterIsInstance<TextPromptPart>().joinToString("\n") { it.text }
        if (text.isEmpty()) return null
        return ResponseInputItem.ofEasyInputMessage(
            EasyInputMessage.builder().role(EasyInputMessage.Role.USER).content(text).build()
        )
    }

    val contentParts = parts.mapNotNull { it.toUserContent() }
    if (contentParts.isEmpty()) return null
    return ResponseInputItem.ofMessage(
        ResponseInputItem.Message.builder()
            .role(ResponseInputItem.Message.Role.USER)
            .content(contentParts)
            .build()
    )
}

/**
 * Maps a single request-side part into a user-message content part, or `null` if it can't be
 * represented in a Responses user message (non-image binary/file, or a tool part with no place on
 * the user role).
 */
private fun PromptPart.toUserContent(): ResponseInputContent? =
    when (this) {
        is TextPromptPart -> ResponseInputContent.ofInputText(
            ResponseInputText.builder().text(text).build()
        )

        is DataPromptPart -> {
            if (mimeType.startsWith("image/")) {
                val dataUrl = "data:$mimeType;base64,${Base64.getEncoder().encodeToString(data)}"
                imageContent(dataUrl)
            } else {
                logger.warn("Dropping non-image DataPromptPart (mime={}) — Responses input accepts only images", mimeType)
                null
            }
        }

        is FileDataPromptPart -> {
            if (uri.startsWith("http://") || uri.startsWith("https://")) {
                imageContent(uri)
            } else {
                logger.warn("Dropping FileDataPromptPart with non-http(s) uri '{}' — needs a URL or data URL", uri)
                null
            }
        }

        is ToolCallPromptPart, is ToolResultPromptPart -> null
    }

private fun imageContent(url: String): ResponseInputContent =
    ResponseInputContent.ofInputImage(
        ResponseInputImage.builder()
            .imageUrl(url)
            .detail(ResponseInputImage.Detail.AUTO)
            .build()
    )

/**
 * Provider-neutral request dump for telemetry — identical to how gemini builds `requestMessages`
 * from the outgoing contents. Uses the core [PromptContent.toMessage].
 */
internal fun List<PromptContent>.toMessages(): List<PromptMessage> = withCallsLast().map { it.toMessage() }
