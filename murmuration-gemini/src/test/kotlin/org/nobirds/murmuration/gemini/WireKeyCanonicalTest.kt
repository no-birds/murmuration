package org.nobirds.murmuration.gemini

import org.nobirds.murmuration.prompt.CallIds
import com.google.genai.types.Candidate
import com.google.genai.Client
import com.google.genai.types.GenerateContentConfig
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptCachingProvider
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptMessage
import org.nobirds.murmuration.prompt.MusicPromptSettings
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.ToolCall
import org.nobirds.murmuration.prompt.ToolCallPromptPart
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.schema.SchemaInfo
import org.nobirds.murmuration.tooling.Instrument
import org.nobirds.murmuration.tooling.InstrumentParam
import com.google.genai.types.Content
import com.google.genai.types.FunctionCall
import com.google.genai.types.GenerateContentResponse
import com.google.genai.types.Part
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import java.util.Base64
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

class WireKeyCanonicalTest {

    private fun obj(vararg p: Pair<String, JsonElement>) = JsonObject(mapOf(*p))

    @Test
    fun `toCanonical strips prefix on nested objects and arrays`() {
        val wire = obj(
            "~aname" to JsonPrimitive("x"),
            "plain" to obj("~Binner" to JsonArray(listOf(obj("~0deep" to JsonPrimitive(1)), JsonPrimitive("s")))),
        )
        val expected = obj(
            "name" to JsonPrimitive("x"),
            "plain" to obj("inner" to JsonArray(listOf(obj("deep" to JsonPrimitive(1)), JsonPrimitive("s")))),
        )
        assertEquals(expected, wire.toCanonical())
    }

    @Test
    fun `toCanonical leaves keys without prefix and too short markers untouched`() {
        val canonical = obj("name" to JsonPrimitive("x"), "~" to JsonPrimitive(1), "~a" to JsonPrimitive(2))
        assertEquals(canonical, canonical.toCanonical())
        assertEquals(canonical.toCanonical(), canonical.toCanonical().toCanonical())
    }

    private fun wireCall(signature: ByteArray? = null): Part {
        val b = Part.builder().functionCall(
            FunctionCall.builder().name("t").args(
                mapOf("~aname" to "x", "nested" to mapOf("~bk" to listOf(mapOf("~cz" to 1L)))),
            ).build(),
        )
        signature?.let { b.thoughtSignature(it) }
        return b.build()
    }

    private val canonicalArgs: Map<String, JsonElement> = mapOf(
        "name" to JsonPrimitive("x"),
        "nested" to obj("k" to JsonArray(listOf(obj("z" to JsonPrimitive(1))))),
    )

    @Test
    fun `stream path gives canonical ToolCall and keeps thoughtSignature`() {
        val sig = "sig".toByteArray()
        val part = wireCall(sig)
        val response = GenerateContentResponse.builder()
            .candidates(listOf(Candidate.builder().content(Content.builder().parts(listOf(part)).build()).build()))
            .build()
        val event = response.allParts().mapNotNull { it.toEventOrNull() }.single()
        val call = assertIs<MessagePart.ToolCall>(assertIs<PartReceived>(event).part)
        assertEquals("t", call.name)
        assertEquals(canonicalArgs, call.args)
        assertEquals(Base64.getEncoder().encodeToString(sig), call.thoughtSignature)
    }

    @Test
    fun `non-stream path gives the same canonical ToolCall`() {
        val call = assertIs<MessagePart.ToolCall>(wireCall().toMessagePart())
        assertEquals(canonicalArgs, call.args)
    }
}

class RequestTelemetryCanonicalTest {

    private fun history(args: Map<String, Any>) = listOf(
        Content.builder().role("model").parts(
            listOf(
                Part.builder().text("hi").build(),
                Part.builder().functionCall(FunctionCall.builder().name("t").args(args).build()).build(),
            ),
        ).build(),
    )

    @Test
    fun `without wire names output equals the plain mapping`() {
        val msgs = history(mapOf("a" to "x", "b" to mapOf("c" to listOf(1L)))).toMessages().single()
        val call = msgs.parts.filterIsInstance<MessagePart.ToolCall>().single()
        assertEquals(
            mapOf(
                "a" to JsonPrimitive("x"),
                "b" to JsonObject(mapOf("c" to JsonArray(listOf(JsonPrimitive(1))))),
            ),
            call.args,
        )
        assertEquals(2, msgs.parts.size)
    }

    @Test
    fun `wire keys in history are written to telemetry as canonical names`() {
        val msgs = history(mapOf("~aa" to "x", "b" to mapOf("~bc" to 2L))).toMessages().single()
        val call = msgs.parts.filterIsInstance<MessagePart.ToolCall>().single()
        assertEquals(
            mapOf("a" to JsonPrimitive("x"), "b" to JsonObject(mapOf("c" to JsonPrimitive(2)))),
            call.args,
        )
    }

    @Serializable data class Inner(@SchemaInfo("i") val id: Int, @SchemaInfo("n") val name: String)

    @Instrument
    class Tools {
        @Instrument("d") fun act(@InstrumentParam("p") p: Inner, @InstrumentParam("q") q: String): String = ""
    }

    private class CapturingStrategy : GeminiResponseStrategy {
        var contents: List<Content> = emptyList()
        var requestMessages: List<PromptMessage> = emptyList()

        override fun execute(
            prompt: Prompt,
            modelName: String,
            contents: List<Content>,
            config: GenerateContentConfig,
            cacheResult: PromptCachingProvider.CacheResult?,
            requestMessages: List<PromptMessage>,
            requestedModelName: String,
        ): Flow<PromptExecutionEvent> = flow {
            this@CapturingStrategy.contents = contents
            this@CapturingStrategy.requestMessages = requestMessages
            emit(PromptExecutionEvent.Completed())
        }
    }

    private val canonicalArgs = mapOf<String, JsonElement>(
        "p" to JsonObject(mapOf("id" to JsonPrimitive(1), "name" to JsonPrimitive("a"))),
        "q" to JsonPrimitive("x"),
    )

    private fun historyContent() = listOf(
        PromptContent(PromptContentType.USER, listOf(TextPromptPart("go"))),
        PromptContent(PromptContentType.MODEL, listOf(ToolCallPromptPart(ToolCall("act", canonicalArgs, callId = CallIds.mint())))),
    )

    private fun run(prompt: Prompt, layout: GeminiWireKeyLayout): CapturingStrategy {
        val strategy = CapturingStrategy()
        val client = GeminiClient(Client.builder().apiKey("test-key-not-used").build(), layout, responseStrategy = strategy)
        runBlocking { client.execute(prompt).collect {} }
        return strategy
    }

    private fun callOf(contents: List<Content>): Map<String, Any> =
        contents.flatMap { it.parts().orElse(emptyList()) }
            .mapNotNull { it.functionCall().orElse(null) }.single().args().get()

    @Test
    fun `with a table the genai request carries wire names and telemetry canonical names`() {
        val tools = Tools()
        val layout = WireLayoutTestSupport.layoutOf(WireLayoutTestSupport.canonicalSchemas(WireLayoutTestSupport.instrumentsOf(tools)))
        val prompt = buildPrompt {
            llm = GeminiLlms.Vertex.VERTEX_GEMINI_3_FLASH
            settings = WireLayoutTestSupport.settingsOf(tools)
            content(historyContent())
        }
        val run = run(prompt, layout)

        assertTrue(callOf(run.contents).keys.all { it.startsWith("~") }, "request: ${callOf(run.contents).keys}")
        val call = run.requestMessages.flatMap { it.parts }.filterIsInstance<MessagePart.ToolCall>().single()
        assertEquals(canonicalArgs, call.args)
    }

    @Test
    fun `with NONE request and telemetry are both canonical`() {
        val prompt = buildPrompt {
            llm = GeminiLlms.Vertex.VERTEX_GEMINI_3_FLASH
            settings = WireLayoutTestSupport.settingsOf(Tools())
            content(historyContent())
        }
        val run = run(prompt, GeminiWireKeyLayout.NONE)

        assertEquals(setOf("p", "q"), callOf(run.contents).keys)
        assertEquals(canonicalArgs, run.requestMessages.flatMap { it.parts }.filterIsInstance<MessagePart.ToolCall>().single().args)
    }

    @Test
    fun `music path has no tools - request and telemetry stay canonical even with a table`() {
        val layout = WireLayoutTestSupport.layoutOf(WireLayoutTestSupport.canonicalSchemas(WireLayoutTestSupport.instrumentsOf(Tools())))
        val prompt = buildPrompt {
            llm = GeminiLlms.Vertex.VERTEX_LYRIA_3_PRO
            settings = MusicPromptSettings()
            content(historyContent())
        }
        val run = run(prompt, layout)

        assertEquals(setOf("p", "q"), callOf(run.contents).keys)
        assertEquals(canonicalArgs, run.requestMessages.flatMap { it.parts }.filterIsInstance<MessagePart.ToolCall>().single().args)
    }
}
