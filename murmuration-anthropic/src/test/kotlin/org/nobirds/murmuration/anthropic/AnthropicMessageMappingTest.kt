package org.nobirds.murmuration.anthropic

import com.anthropic.models.messages.ContentBlockParam
import com.anthropic.models.messages.MessageParam
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.CallId
import org.nobirds.murmuration.prompt.DataPromptPart
import org.nobirds.murmuration.prompt.FileDataPromptPart
import org.nobirds.murmuration.prompt.InvalidRequestException
import org.nobirds.murmuration.prompt.PromptCacheStrategy
import org.nobirds.murmuration.prompt.PartKind
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptPart
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.ToolCall
import org.nobirds.murmuration.prompt.ToolCallPromptPart
import org.nobirds.murmuration.prompt.ToolResultPromptPart
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.time.Duration.Companion.hours

class AnthropicMessageMappingTest {

    private fun prompt(vararg content: PromptContent): Prompt =
        Prompt(AnthropicLlms.findModel("anthropic", "claude-sonnet-5-5"), null, content.toList())

    private fun content(type: PromptContentType, vararg parts: PromptPart) = PromptContent(type, parts.toList())

    private fun text(text: String) = TextPromptPart(text)

    private fun thought(text: String, carrier: String?) = TextPromptPart(text, thoughtSignature = carrier, kind = PartKind.THOUGHT)

    private fun call(id: String, name: String) =
        ToolCallPromptPart(ToolCall(name, mapOf("n" to JsonPrimitive(1), "s" to JsonPrimitive("x")), callId = CallId(id)))

    private fun result(id: String, name: String, value: String) = ToolResultPromptPart(name, JsonPrimitive(value), CallId(id))

    /** One readable line per block: the exact sequence is what the API sees. */
    private fun describe(block: ContentBlockParam): String = when {
        block.isText() -> "text:${block.asText().text()}"
        block.isThinking() -> "thinking:${block.asThinking().thinking()}|${block.asThinking().signature()}"
        block.isRedactedThinking() -> "redacted:${block.asRedactedThinking().data()}"
        block.isToolUse() -> "tool_use:${block.asToolUse().id()}|${block.asToolUse().name()}"
        block.isToolResult() -> "tool_result:${block.asToolResult().toolUseId()}|${block.asToolResult().content().get().asString()}"
        else -> "other"
    }

    private fun MessageParam.blocks(): List<String> = content().asBlockParams().map(::describe)

    @Test
    fun `canonical history maps to the exact block sequence`() {
        val mapped = prompt(
            content(PromptContentType.SYSTEM, text("rules"), text("world")),
            content(PromptContentType.USER, text("go")),
            content(
                PromptContentType.MODEL,
                thought("t1", "SIG1"),
                text("let me look"),
                call("toolu_a", "look"),
                thought("t2", "SIG2"),
                thought("", AnthropicThoughtCarrier.encodeRedacted("OPAQUE")),
                call("toolu_b", "roll"),
            ),
            // Results arrive in the reverse order of the calls: the history order is kept as is.
            content(PromptContentType.TOOL, result("toolu_b", "roll", "17"), result("toolu_a", "look", "a door")),
        ).toAnthropicMessages()

        assertEquals(listOf("rules", "world"), mapped.system.map { it.text() })
        assertEquals(3, mapped.messages.size)
        assertEquals(MessageParam.Role.USER, mapped.messages[0].role())
        assertEquals(listOf("text:go"), mapped.messages[0].blocks())
        assertEquals(MessageParam.Role.ASSISTANT, mapped.messages[1].role())
        assertEquals(
            listOf(
                "thinking:t1|SIG1",
                "text:let me look",
                "tool_use:toolu_a|look",
                "thinking:t2|SIG2",
                "redacted:OPAQUE",
                "tool_use:toolu_b|roll",
            ),
            mapped.messages[1].blocks(),
        )
        assertEquals(MessageParam.Role.USER, mapped.messages[2].role())
        assertEquals(listOf("tool_result:toolu_b|\"17\"", "tool_result:toolu_a|\"a door\""), mapped.messages[2].blocks())
    }

    @Test
    fun `tool_use input carries the call arguments`() {
        val mapped = prompt(
            content(PromptContentType.MODEL, call("toolu_a", "look")),
            content(PromptContentType.TOOL, result("toolu_a", "look", "ok")),
        ).toAnthropicMessages()

        val input = mapped.messages[0].content().asBlockParams().single().asToolUse().input()._additionalProperties()
        assertEquals(setOf("n", "s"), input.keys)
        assertEquals(1L, input.getValue("n").asNumber().get())
        assertEquals("x", input.getValue("s").asString().get())
    }

    @Test
    fun `one text part of a model message is one text block`() {
        val mapped = prompt(
            content(PromptContentType.USER, text("go")),
            content(PromptContentType.MODEL, text("Hello world")),
        ).toAnthropicMessages()

        assertEquals(listOf("text:Hello world"), mapped.messages[1].blocks())
    }

    @Test
    fun `thought without a carrier is omitted, not replayed as text`() {
        val mapped = prompt(
            content(PromptContentType.USER, text("go")),
            content(PromptContentType.MODEL, thought("foreign thought", null), thought("empty carrier", ""), text("answer")),
        ).toAnthropicMessages()

        assertEquals(listOf("text:answer"), mapped.messages[1].blocks())
    }

    @Test
    fun `a model message holding only an unsigned thought disappears and neighbours merge`() {
        val mapped = prompt(
            content(PromptContentType.USER, text("one")),
            content(PromptContentType.MODEL, thought("lost", null)),
            content(PromptContentType.USER, text("two")),
        ).toAnthropicMessages()

        assertEquals(1, mapped.messages.size)
        assertEquals(listOf("text:one", "text:two"), mapped.messages[0].blocks())
    }

    @Test
    fun `tool results and the following user message merge with results first`() {
        val mapped = prompt(
            content(PromptContentType.MODEL, call("toolu_a", "look")),
            content(PromptContentType.USER, text("also, hurry")),
            content(PromptContentType.TOOL, result("toolu_a", "look", "ok")),
        ).toAnthropicMessages()

        assertEquals(2, mapped.messages.size)
        assertEquals(listOf("tool_result:toolu_a|\"ok\"", "text:also, hurry"), mapped.messages[1].blocks())
    }

    @Test
    fun `toRequest carries system blocks and messages`() {
        val params = prompt(
            content(PromptContentType.SYSTEM, text("a"), text("b")),
            content(PromptContentType.USER, text("go")),
        ).toRequest()

        assertEquals(listOf("a", "b"), params.system().get().asTextBlockParams().map { it.text() })
        assertEquals(1, params.messages().size)
    }

    // ---- refusals and pairing, all before the network ----

    private fun refused(vararg content: PromptContent): InvalidRequestException =
        assertFailsWith<InvalidRequestException> { prompt(*content).toAnthropicMessages() }

    @Test
    fun `a legacy call id is refused on a call and on a result`() {
        refused(content(PromptContentType.MODEL, call("", "look")))
        refused(
            content(PromptContentType.MODEL, call("toolu_a", "look")),
            content(PromptContentType.TOOL, result("", "look", "ok")),
        )
    }

    @Test
    fun `a result with no call in the preceding model turn is refused`() {
        refused(content(PromptContentType.USER, text("go")), content(PromptContentType.TOOL, result("toolu_x", "look", "ok")))
        refused(
            content(PromptContentType.MODEL, call("toolu_a", "look")),
            content(PromptContentType.TOOL, result("toolu_a", "look", "ok"), result("toolu_x", "roll", "1")),
        )
    }

    @Test
    fun `a call with no result gets an error result in the next user message, and in a message of its own at the end`() {
        val mapped = prompt(
            content(PromptContentType.USER, text("go")),
            content(PromptContentType.MODEL, call("toolu_a", "look"), call("toolu_b", "roll")),
            content(PromptContentType.TOOL, result("toolu_b", "roll", "17")),
        ).toAnthropicMessages()
        val results = mapped.messages[2].content().asBlockParams().map { it.asToolResult() }
        assertEquals(listOf("toolu_b", "toolu_a"), results.map { it.toolUseId() })
        assertEquals(listOf(false, true), results.map { it.isError().orElse(false) })

        val tail = prompt(content(PromptContentType.MODEL, call("toolu_a", "look"))).toAnthropicMessages()
        assertEquals(MessageParam.Role.USER, tail.messages.last().role())
        assertEquals(true, tail.messages.last().content().asBlockParams().single().asToolResult().isError().get())
    }

    @Test
    fun `a PDF, any non-image data and any file part are refused, an image goes through`() {
        refused(content(PromptContentType.USER, DataPromptPart(byteArrayOf(1, 2), "application/pdf")))
        refused(content(PromptContentType.USER, FileDataPromptPart("gs://bucket/book.pdf", mimeType = "application/pdf")))
        refused(content(PromptContentType.USER, FileDataPromptPart("gs://bucket/pic.png", mimeType = "image/png")))
        val ok = prompt(content(PromptContentType.USER, text("look"), DataPromptPart(byteArrayOf(1, 2), "image/png"))).toAnthropicMessages()
        assertEquals(2, ok.messages.single().content().asBlockParams().size)
        assertEquals(true, ok.messages.single().content().asBlockParams()[1].isImage())
    }

    @Test
    fun `the cache strategy of each part rides with its block`() {
        val hour = PromptCacheStrategy.ttl(1.hours)
        val mapped = prompt(
            content(PromptContentType.SYSTEM, TextPromptPart("rules", cacheStrategy = hour)),
            content(PromptContentType.USER, TextPromptPart("go", cacheStrategy = PromptCacheStrategy.PIPELINE)),
        ).toAnthropicMessages()
        assertEquals(listOf(hour), mapped.systemCache)
        assertEquals(listOf(listOf(PromptCacheStrategy.PIPELINE)), mapped.messageCache)
    }
}
