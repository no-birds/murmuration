package org.nobirds.murmuration.openai

import org.nobirds.murmuration.prompt.CallIds
import com.openai.models.responses.EasyInputMessage
import com.openai.models.responses.ResponseInputItem
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.DataPromptPart
import org.nobirds.murmuration.prompt.FileDataPromptPart
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.ToolCall
import org.nobirds.murmuration.prompt.ToolCallPromptPart
import org.nobirds.murmuration.prompt.ToolResultPromptPart
import kotlin.jvm.optionals.getOrNull
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Unit tests for [List<PromptContent>.toResponseInput] — the provider-neutral message history ->
 * Azure OpenAI Responses input-item mapper. Asserts OUR role/part mapping, synthetic call_id
 * assignment and reasoning-item replay, not the SDK's own (de)serialisation.
 */
class OpenAiMessageMappingTest {

    @Test
    fun `system parts are joined into a single developer message`() {
        val items = listOf(
            PromptContent(PromptContentType.SYSTEM, listOf(TextPromptPart("Rule one"), TextPromptPart("Rule two"))),
        ).toResponseInput()

        assertEquals(1, items.size)
        val msg = items[0].easyInputMessage().getOrNull()!!
        assertEquals(EasyInputMessage.Role.DEVELOPER, msg.role())
        assertEquals("Rule one\n\nRule two", msg.content().asTextInput())
    }

    @Test
    fun `user text becomes a plain-string user message`() {
        val items = listOf(
            PromptContent(PromptContentType.USER, listOf(TextPromptPart("Hello"))),
        ).toResponseInput()

        assertEquals(1, items.size)
        val msg = items[0].easyInputMessage().getOrNull()!!
        assertEquals(EasyInputMessage.Role.USER, msg.role())
        assertEquals("Hello", msg.content().asTextInput())
    }

    @Test
    fun `user image data part promotes the message to a data-URL input image`() {
        val items = listOf(
            PromptContent(
                PromptContentType.USER,
                listOf(TextPromptPart("look"), DataPromptPart(byteArrayOf(1, 2, 3), "image/png")),
            ),
        ).toResponseInput()

        val content = items[0].message().getOrNull()!!.content()
        assertEquals(2, content.size)
        assertTrue(content[0].isInputText())
        assertTrue(content[1].isInputImage())
        assertTrue(content[1].asInputImage().imageUrl().getOrNull()!!.startsWith("data:image/png;base64,"))
    }

    @Test
    fun `user file data part with an http uri becomes an input image url`() {
        val items = listOf(
            PromptContent(PromptContentType.USER, listOf(FileDataPromptPart("https://example.com/a.png"))),
        ).toResponseInput()

        val content = items[0].message().getOrNull()!!.content()
        assertEquals(1, content.size)
        assertEquals("https://example.com/a.png", content[0].asInputImage().imageUrl().getOrNull())
    }

    @Test
    fun `model tool calls become function_call items with synthetic call ids`() {
        val call1 = ToolCall("get_weather", mapOf("city" to JsonPrimitive("Paris")), callId = CallIds.mint())
        val call2 = ToolCall("get_time", mapOf("tz" to JsonPrimitive("UTC")), callId = CallIds.mint())
        val items = listOf(
            PromptContent(
                PromptContentType.MODEL,
                listOf(TextPromptPart("Let me check"), ToolCallPromptPart(call1), ToolCallPromptPart(call2)),
            ),
        ).toResponseInput()

        // assistant text message + two function calls
        assertEquals(3, items.size)
        val assistant = items[0].easyInputMessage().getOrNull()!!
        assertEquals(EasyInputMessage.Role.ASSISTANT, assistant.role())
        assertEquals("Let me check", assistant.content().asTextInput())

        val fc0 = items[1].functionCall().getOrNull()!!
        assertEquals("call_0", fc0.callId())
        assertEquals("get_weather", fc0.name())
        val fc1 = items[2].functionCall().getOrNull()!!
        assertEquals("call_1", fc1.callId())
        assertEquals("get_time", fc1.name())
    }

    @Test
    fun `tool results become function_call_output matched positionally to the preceding calls`() {
        val items = listOf(
            PromptContent(
                PromptContentType.MODEL,
                listOf(ToolCallPromptPart(ToolCall("get_weather", emptyMap(), callId = CallIds.mint())), ToolCallPromptPart(ToolCall("get_time", emptyMap(), callId = CallIds.mint()))),
            ),
            PromptContent(
                PromptContentType.TOOL,
                listOf(
                    ToolResultPromptPart("get_weather", JsonPrimitive("sunny"), callId = CallIds.mint()),
                    ToolResultPromptPart("get_time", JsonPrimitive("noon"), callId = CallIds.mint()),
                ),
            ),
        ).toResponseInput()

        // 2 function calls + 2 outputs
        assertEquals(4, items.size)
        val out0 = items[2].functionCallOutput().getOrNull()!!
        val out1 = items[3].functionCallOutput().getOrNull()!!
        assertEquals("call_0", out0.callId().get())
        assertEquals(JsonPrimitive("sunny").toString(), out0.output().asString())
        assertEquals("call_1", out1.callId().get())
        assertEquals(JsonPrimitive("noon").toString(), out1.output().asString())
    }

    @Test
    fun `orphan tool result without a preceding model call gets a synthetic orphan id`() {
        val items = listOf(
            PromptContent(PromptContentType.TOOL, listOf(ToolResultPromptPart("mystery", JsonPrimitive("ok"), callId = CallIds.mint()))),
        ).toResponseInput()

        assertEquals(1, items.size)
        assertEquals("call_orphan_0", items[0].functionCallOutput().getOrNull()!!.callId().get())
    }

    @Test
    fun `a reasoning signature on a tool call replays as a reasoning item immediately before it`() {
        val signature = encodeReasoningSignature("rs_abc", "enc-blob")
        val items = listOf(
            PromptContent(
                PromptContentType.MODEL,
                listOf(
                    TextPromptPart("Let me check"),
                    ToolCallPromptPart(ToolCall("get_weather", emptyMap(), thoughtSignature = signature, callId = CallIds.mint())),
                    ToolCallPromptPart(ToolCall("get_time", emptyMap(), callId = CallIds.mint())),
                ),
            ),
        ).toResponseInput()

        // assistant text, reasoning, its paired call, then the unsigned call
        assertEquals(4, items.size)
        assertTrue(items[0].isEasyInputMessage())
        val reasoning = items[1].reasoning().getOrNull()!!
        assertEquals("rs_abc", reasoning.id())
        assertEquals("enc-blob", reasoning.encryptedContent().getOrNull())
        assertTrue(items[2].isFunctionCall())
        assertEquals("get_weather", items[2].functionCall().getOrNull()!!.name())
        assertTrue(items[3].isFunctionCall())
    }

    @Test
    fun `a fallback reasoning signature on a text part replays as a reasoning item`() {
        val signature = encodeReasoningSignature("rs_txt", "enc2")
        val items = listOf(
            PromptContent(
                PromptContentType.MODEL,
                listOf(TextPromptPart("", signature), TextPromptPart("The answer.")),
            ),
        ).toResponseInput()

        assertEquals(2, items.size)
        assertEquals("rs_txt", items[0].reasoning().getOrNull()!!.id())
        assertTrue(items[1].isEasyInputMessage())
    }

    @Test
    fun `a reasoning signature without encrypted content is dropped (cannot be replayed)`() {
        val signature = encodeReasoningSignature("rs_abc", null)
        val items = listOf(
            PromptContent(
                PromptContentType.MODEL,
                listOf(
                    TextPromptPart("", signature),
                    ToolCallPromptPart(ToolCall("get_weather", emptyMap(), thoughtSignature = signature, callId = CallIds.mint())),
                ),
            ),
        ).toResponseInput()

        // only the function call survives — a reasoning item is unusable without encrypted_content
        assertEquals(1, items.size)
        assertTrue(items[0].isFunctionCall())
    }

    @Test
    fun `model turn in arrival order is laid out with calls last in input and in the trace`() {
        fun call(n: String) = ToolCallPromptPart(ToolCall(n, emptyMap(), null, CallIds.mint()))
        val contents = listOf(
            PromptContent(
                PromptContentType.MODEL,
                listOf(TextPromptPart("t1"), call("c1"), TextPromptPart("t2"), call("c2")),
            ),
        )

        val kinds = contents.toResponseInput().map {
            when {
                it.isFunctionCall() -> "call:" + it.asFunctionCall().name()
                else -> "other"
            }
        }
        assertEquals(listOf("call:c1", "call:c2"), kinds.takeLast(2))
        assertTrue(kinds.dropLast(2).none { it.startsWith("call:") })

        val traceParts = contents.toMessages().single().parts
        assertEquals(
            listOf("t1", "t2", "c1", "c2"),
            traceParts.map { if (it is MessagePart.Text) it.text else (it as MessagePart.ToolCall).name },
        )
    }
}
