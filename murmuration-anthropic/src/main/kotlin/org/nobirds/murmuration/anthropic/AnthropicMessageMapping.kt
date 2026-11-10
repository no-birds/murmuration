package org.nobirds.murmuration.anthropic

import com.anthropic.core.JsonValue
import com.anthropic.models.messages.Base64ImageSource
import com.anthropic.models.messages.ContentBlockParam
import com.anthropic.models.messages.ImageBlockParam
import com.anthropic.models.messages.MessageParam
import com.anthropic.models.messages.TextBlockParam
import com.anthropic.models.messages.ThinkingBlockParam
import com.anthropic.models.messages.ToolResultBlockParam
import com.anthropic.models.messages.ToolUseBlockParam
import kotlinx.serialization.json.JsonElement
import org.nobirds.murmuration.prompt.CallId
import org.nobirds.murmuration.prompt.DataPromptPart
import org.nobirds.murmuration.prompt.FileDataPromptPart
import org.nobirds.murmuration.prompt.InvalidRequestException
import org.nobirds.murmuration.prompt.PartKind
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptCacheStrategy
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptPart
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.ToolCallPromptPart
import org.nobirds.murmuration.prompt.ToolResultPromptPart
import org.nobirds.murmuration.prompt.toAny
import java.util.Base64

/** The prompt history as the Messages API wants it: the `system` blocks and the alternating `messages`. */
internal data class AnthropicMessages(
    val system: List<TextBlockParam>,
    val messages: List<MessageParam>,
    /** Cache wish of the prompt part each [system] block came from, one per block (input of the cache materializer). */
    val systemCache: List<PromptCacheStrategy>,
    /** Same for [messages]: one list per message, one entry per block in the block order of the message. */
    val messageCache: List<List<PromptCacheStrategy>>,
)

/**
 * Converts the history of [Prompt] into [AnthropicMessages]. Cache marks, tool schemas and thinking
 * configuration are not part of this mapping.
 *
 * - `SYSTEM` parts → `system` text blocks, one per non-empty text part, in order.
 * - `USER` → user message: text blocks, image blocks for `image/…` data parts.
 * - `MODEL` → assistant message; **blocks keep the order the parts came in** (the kind of a text part is read from
 *   [TextPromptPart.kind]). A THOUGHT part with a carrier becomes `thinking(text, signature)` or `redacted_thinking(data)`
 *   byte-for-byte (see [AnthropicThoughtCarrier]); a THOUGHT part without a carrier (a Gemini or foreign thought) is
 *   omitted — never replayed as text. A tool call → `tool_use` with `id = CallId`.
 * - `TOOL` → user message of `tool_result` blocks with `tool_use_id = CallId`.
 *
 * Adjacent messages of one role are merged into one (the API requires alternation), and within a user message all
 * `tool_result` blocks come first, as the API demands. A message left with no blocks is dropped.
 *
 * Before the network (all [InvalidRequestException], not retried): a [CallId.LEGACY] id on a call or a result; a result
 * whose call is not in the preceding model turn; any file part, and any data part that is not an image (a PDF is
 * explicitly unsupported — it must not silently drop out of the prompt). The other direction is tolerated: a call with no
 * result in the next user message gets a synthesized `tool_result` with `is_error = true`.
 */
internal fun Prompt.toAnthropicMessages(): AnthropicMessages {
    val system = mutableListOf<TextBlockParam>()
    val systemCache = mutableListOf<PromptCacheStrategy>()
    val turns = mutableListOf<Turn>()

    fun blocks(item: PromptContent, build: (PromptPart) -> ContentBlockParam?): List<SourcedBlock> =
        item.parts.mapNotNull { part -> build(part)?.let { SourcedBlock(it, part.cacheStrategy) } }

    content.forEach { item ->
        item.parts.forEach { part -> requireMappable(part, item.type) }
        when (item.type) {
            PromptContentType.SYSTEM -> item.parts.forEach { part ->
                if (part is TextPromptPart && part.text.isNotEmpty()) {
                    system += TextBlockParam.builder().text(part.text).build()
                    systemCache += part.cacheStrategy
                }
            }

            PromptContentType.USER -> turns.append(MessageParam.Role.USER, blocks(item, ::userBlock))
            PromptContentType.MODEL -> turns.append(MessageParam.Role.ASSISTANT, blocks(item, ::modelBlock))
            PromptContentType.TOOL -> turns.append(MessageParam.Role.USER, blocks(item, ::toolResultBlock))
        }
    }
    turns.pairToolCalls(this)

    val messages = mutableListOf<MessageParam>()
    val messageCache = mutableListOf<List<PromptCacheStrategy>>()
    for (turn in turns) {
        // Stable partition: tool_result blocks first, the rest keeps its order; a block keeps its own cache data.
        val (results, rest) = turn.blocks.partition { it.block.isToolResult() }
        val ordered = results + rest
        messages += MessageParam.builder().role(turn.role).contentOfBlockParams(ordered.map { it.block }).build()
        messageCache += ordered.map { it.cache }
    }
    return AnthropicMessages(system, messages, systemCache, messageCache)
}

private fun Prompt.invalid(message: String): InvalidRequestException =
    InvalidRequestException(this, httpCode = null, message = message, cause = null)

/** A part the Messages API cannot take is refused before the network, not dropped: the model must never answer a prompt that silently lost a part. */
private fun Prompt.requireMappable(part: PromptPart, type: PromptContentType) {
    when (part) {
        is FileDataPromptPart -> throw invalid(
            "Anthropic provider does not support file parts (uri=${part.uri}, mime=${part.mimeType}): only text and images are sent to Claude"
        )

        is DataPromptPart -> {
            if (!part.mimeType.startsWith("image/")) {
                throw invalid("Anthropic provider does not support data parts of type ${part.mimeType}: only text and images are sent to Claude")
            }
            if (type != PromptContentType.USER) throw invalid("An image can only be sent in a user message, not in a ${type.name} content")
        }

        is ToolCallPromptPart -> if (part.call.callId.isLegacy) throw invalid("Tool call '${part.call.name}' has a legacy call id and cannot be sent")
        is ToolResultPromptPart -> if (part.callId.isLegacy) throw invalid("Tool result '${part.name}' has a legacy call id and cannot be sent")
        is TextPromptPart -> Unit
    }
}

/**
 * Pairs `tool_use` with `tool_result` over the assembled turns, as the API demands: the results of the calls of a model
 * turn are the leading blocks of the user turn right after it. A result with no call in the preceding model turn is an
 * [InvalidRequestException]; a call with no result gets a synthesized error result (appended to the next user turn, or a
 * user turn of its own when the history ends there).
 */
private fun MutableList<Turn>.pairToolCalls(prompt: Prompt) {
    var index = 0
    while (index < size) {
        val turn = this[index]
        val calledIds = if (turn.role == MessageParam.Role.ASSISTANT) turn.blocks.mapNotNull { it.block.toolUseIdOrNull() } else emptyList()
        val next = getOrNull(index + 1)?.takeIf { it.role == MessageParam.Role.USER }
        val resultIds = next?.blocks?.mapNotNull { it.block.toolResultIdOrNull() }.orEmpty()
        resultIds.firstOrNull { it !in calledIds }?.let {
            throw prompt.invalid("Tool result for call '$it' has no matching call in the preceding model turn")
        }
        val missing = calledIds.filter { it !in resultIds }
        if (missing.isNotEmpty()) {
            val synthesized = missing.map { SourcedBlock(missingResultBlock(it), PromptCacheStrategy.NO_CACHE) }
            if (next != null) next.blocks += synthesized else add(index + 1, Turn(MessageParam.Role.USER, synthesized.toMutableList()))
        }
        index++
    }
    // A result in the very first turn, or after a user turn, has no call to pair with.
    firstOrNull()?.takeIf { it.role == MessageParam.Role.USER }?.blocks?.firstNotNullOfOrNull { it.block.toolResultIdOrNull() }?.let {
        throw prompt.invalid("Tool result for call '$it' has no matching call in the preceding model turn")
    }
}

private fun ContentBlockParam.toolUseIdOrNull(): String? = if (isToolUse()) asToolUse().id() else null

private fun ContentBlockParam.toolResultIdOrNull(): String? = if (isToolResult()) asToolResult().toolUseId() else null

private fun missingResultBlock(callId: String): ContentBlockParam = ContentBlockParam.ofToolResult(
    ToolResultBlockParam.builder().toolUseId(callId).content("No result was recorded for this tool call.").isError(true).build()
)

/** A built block together with the cache strategy of the prompt part it was built from. */
private class SourcedBlock(val block: ContentBlockParam, val cache: PromptCacheStrategy)

private class Turn(val role: MessageParam.Role, val blocks: MutableList<SourcedBlock>)

/** Adds [blocks] to the last turn when it has the same role, otherwise starts a new turn; no blocks — no turn. */
private fun MutableList<Turn>.append(role: MessageParam.Role, blocks: List<SourcedBlock>) {
    if (blocks.isEmpty()) return
    val last = lastOrNull()
    if (last != null && last.role == role) last.blocks += blocks else add(Turn(role, blocks.toMutableList()))
}

private fun textBlock(text: String): ContentBlockParam? =
    text.takeIf { it.isNotEmpty() }?.let { ContentBlockParam.ofText(TextBlockParam.builder().text(it).build()) }

private fun userBlock(part: PromptPart): ContentBlockParam? = when (part) {
    is TextPromptPart -> textBlock(part.text)
    is DataPromptPart -> imageBlock(part)
    else -> null
}

private fun imageBlock(part: DataPromptPart): ContentBlockParam {
    val source = Base64ImageSource.builder()
        .data(Base64.getEncoder().encodeToString(part.data))
        .mediaType(Base64ImageSource.MediaType.of(part.mimeType))
        .build()
    return ContentBlockParam.ofImage(ImageBlockParam.builder().source(source).build())
}

private fun modelBlock(part: PromptPart): ContentBlockParam? = when (part) {
    is TextPromptPart -> when (part.kind) {
        PartKind.TEXT -> textBlock(part.text)
        PartKind.THOUGHT -> thoughtBlock(part)
    }

    is ToolCallPromptPart -> ContentBlockParam.ofToolUse(
        ToolUseBlockParam.builder()
            .id(part.call.callId.value)
            .name(part.call.name)
            .input(ToolUseBlockParam.Input.builder().additionalProperties(part.call.args.mapValues { it.value.toJsonValue() }).build())
            .build()
    )

    else -> null
}

private fun thoughtBlock(part: TextPromptPart): ContentBlockParam? =
    when (val carrier = AnthropicThoughtCarrier.decode(part.thoughtSignature)) {
        is AnthropicThoughtCarrier.Signature -> ContentBlockParam.ofThinking(
            ThinkingBlockParam.builder().thinking(part.text).signature(carrier.signature).build()
        )

        is AnthropicThoughtCarrier.Redacted -> ContentBlockParam.ofRedactedThinking(carrier.data)
        null -> null
    }

private fun toolResultBlock(part: PromptPart): ContentBlockParam? =
    (part as? ToolResultPromptPart)?.let {
        ContentBlockParam.ofToolResult(
            ToolResultBlockParam.builder().toolUseId(it.callId.value).content(it.result.toString()).build()
        )
    }

internal fun JsonElement.toJsonValue(): JsonValue = JsonValue.from(toAny())
