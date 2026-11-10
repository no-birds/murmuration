package org.nobirds.murmuration.anthropic

import com.anthropic.core.jsonMapper
import com.anthropic.models.messages.RawMessageStreamEvent
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.CallId
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import kotlin.random.Random
import kotlin.test.assertEquals

/**
 * `tool_use` assembly from `input_json_delta` must not depend on how the wire JSON is cut into `partial_json` pieces.
 * Same args JSON (nested object, quotes, unicode, escapes) is fed with different cut points; the parsed args must be identical.
 */
class AnthropicToolUseDeltaSplitTest {

    private fun wire(json: String): RawMessageStreamEvent = jsonMapper().readValue(json, RawMessageStreamEvent::class.java)

    private fun run(events: List<String>): List<PromptExecutionEvent> {
        val parser = AnthropicStreamParser(requestedModelName = "requested")
        return events.flatMap { parser.accept(wire(it)) }
    }

    /** JSON string literal with every non-ASCII char as \uXXXX, so a cut between surrogates stays valid wire JSON. */
    private fun literal(s: String): String = buildString {
        append('"')
        for (c in s) when {
            c == '"' -> append("\\\"")
            c == '\\' -> append("\\\\")
            c == '\n' -> append("\\n")
            c.code < 0x20 || c.code > 0x7e -> append("\\u%04x".format(c.code))
            else -> append(c)
        }
        append('"')
    }

    private fun blockStart(index: Int, block: String) = """{"type":"content_block_start","index":$index,"content_block":$block}"""
    private fun blockStop(index: Int) = """{"type":"content_block_stop","index":$index}"""
    private fun jsonDelta(index: Int, piece: String) =
        """{"type":"content_block_delta","index":$index,"delta":{"type":"input_json_delta","partial_json":${literal(piece)}}}"""

    private fun toolStart(index: Int, id: String, name: String) = blockStart(index, """{"type":"tool_use","id":"$id","name":"$name","input":{}}""")

    private val tricky = "\u041E\u043D \u0441\u043A\u0430\u0437\u0430\u043B: \"\u043F\u0440\u0438\u0432\u0435\u0442\", \u043F\u0443\u0442\u044C C:\\dir\\n, \u0441\u0442\u0440\u043E\u043A\u04301\n\u0441\u0442\u0440\u043E\u043A\u04302, \u044D\u043C\u043E\u0434\u0437\u0438 \uD83D\uDE00, \u0451"

    private fun argsOf(partId: String, text: String): JsonObject = buildJsonObject {
        put("partId", partId)
        put("part", buildJsonObject { put("type", "STORY"); put("text", text) })
    }

    private fun cutsEvery(json: String, step: Int): List<String> = json.chunked(step)

    private fun cutsRandom(json: String, seed: Int): List<String> {
        val random = Random(seed)
        val pieces = mutableListOf<String>()
        var i = 0
        while (i < json.length) {
            val n = random.nextInt(1, 9)
            pieces += json.substring(i, minOf(json.length, i + n))
            i += n
        }
        return pieces
    }

    /** Cuts right after every backslash (inside an escape) and in the middle of every surrogate pair. */
    private fun cutsInsideEscapesAndSurrogates(json: String): List<String> {
        val pieces = mutableListOf<String>()
        val current = StringBuilder()
        for ((i, c) in json.withIndex()) {
            current.append(c)
            if (c == '\\' || c.isHighSurrogate() || (i > 0 && json[i - 1] == '\\')) {
                pieces += current.toString(); current.clear()
            }
        }
        if (current.isNotEmpty()) pieces += current.toString()
        return pieces
    }

    private fun singleCall(json: String, pieces: List<String>): List<MessagePart> {
        check(pieces.joinToString("") == json)
        val events = listOf(toolStart(0, "toolu_A", "addPart")) + pieces.map { jsonDelta(0, it) } + blockStop(0)
        return run(events).filterIsInstance<PartReceived>().map { it.part }
    }

    @Test
    fun `tool args are identical for any partial_json split`() {
        val expected = argsOf("p1", tricky)
        val json = expected.toString()
        val splits = mapOf(
            "whole" to listOf(json),
            "1 char" to cutsEvery(json, 1),
            "3 chars" to cutsEvery(json, 3),
            "random seed 1" to cutsRandom(json, 1),
            "random seed 42" to cutsRandom(json, 42),
            "random seed 2026" to cutsRandom(json, 2026),
            "inside escapes and surrogates" to cutsInsideEscapesAndSurrogates(json),
        )
        for ((name, pieces) in splits) {
            assertEquals(
                listOf<MessagePart>(MessagePart.ToolCall("addPart", expected, callId = CallId("toolu_A"))),
                singleCall(json, pieces),
                "split: $name",
            )
        }
    }

    @Test
    fun `interleaved parallel tool_use blocks with text and thinking between keep args apart and parts in stop order`() {
        val argsA = argsOf("pA", tricky)
        val argsB = argsOf("pB", "second \"call\"\n\\ \u0451")
        val a = cutsRandom(argsA.toString(), 7)
        val b = cutsEvery(argsB.toString(), 1)

        val events = mutableListOf<String>()
        events += blockStart(0, """{"type":"thinking","thinking":"","signature":""}""")
        events += """{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hm"}}"""
        events += """{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG"}}"""
        events += blockStop(0)
        events += toolStart(1, "toolu_A", "addPart")
        events += toolStart(2, "toolu_B", "addPart")
        // A and B deltas strictly alternate; a text block opens and closes in the middle
        val textAt = 3
        for (i in 0 until maxOf(a.size, b.size)) {
            if (i < a.size) events += jsonDelta(1, a[i])
            if (i < b.size) events += jsonDelta(2, b[i])
            if (i == textAt) {
                events += blockStart(3, """{"type":"text","text":""}""")
                events += """{"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"mid"}}"""
                events += blockStop(3)
            }
        }
        events += blockStop(2)
        events += blockStop(1)

        val parts = run(events).filterIsInstance<PartReceived>().map { it.part }
        assertEquals(
            listOf<MessagePart>(
                MessagePart.Thought("hm", AnthropicThoughtCarrier.encodeSignature("SIG")),
                MessagePart.Text("mid"),
                MessagePart.ToolCall("addPart", argsB, callId = CallId("toolu_B")),
                MessagePart.ToolCall("addPart", argsA, callId = CallId("toolu_A")),
            ),
            parts,
        )
    }
}
