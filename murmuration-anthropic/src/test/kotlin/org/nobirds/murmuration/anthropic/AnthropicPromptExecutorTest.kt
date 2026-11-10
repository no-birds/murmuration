package org.nobirds.murmuration.anthropic

import com.anthropic.models.messages.MessageCreateParams
import com.github.tomakehurst.wiremock.WireMockServer
import com.github.tomakehurst.wiremock.client.WireMock.aResponse
import com.github.tomakehurst.wiremock.client.WireMock.post
import com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo
import com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.CallId
import org.nobirds.murmuration.prompt.InvalidRequestException
import org.nobirds.murmuration.prompt.MaxTokensException
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PolicyViolationException
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptCacheStrategy
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Failed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.ThinkingLevel
import org.nobirds.murmuration.prompt.ToolsPromptSettings
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.UnknownProviderException
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.fileUri
import org.nobirds.murmuration.prompt.system
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.tooling.Instrument
import org.nobirds.murmuration.tooling.buildInstruments
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicReference
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The executor against a WireMock server that plays back a recorded SSE stream. Canonical shape: [stub] the stream,
 * [run] a prompt, then compare the events ([Outcome.events]) or the failure ([Outcome.error]) and read the request
 * body the server saw ([requestBody]).
 */
class AnthropicPromptExecutorTest {

    class PingTool {
        @Instrument("Ping the test tool.")
        fun ping(): String = "pong"
    }

    private lateinit var wireMock: WireMockServer

    @BeforeEach
    fun setUp() {
        wireMock = WireMockServer(wireMockConfig().dynamicPort())
        wireMock.start()
    }

    @AfterEach
    fun tearDown() {
        wireMock.stop()
    }

    private val sonnet = AnthropicLlms.findModel("anthropic", "claude-sonnet-5-5")
    private val haiku = AnthropicLlms.findModel("anthropic", "claude-haiku-5-5")

    private fun executor(auth: AnthropicAuth = AnthropicAuth.ApiKey("test-key"), idleTimeout: Duration = 120.seconds) =
        AnthropicPromptExecutor(
            AnthropicClientFactory.create(auth, baseUrl = "http://localhost:${wireMock.port()}"),
            idleTimeout = idleTimeout,
        )

    private fun sse(vararg events: String): String = events.joinToString("") { json ->
        val type = Json.parseToJsonElement(json).jsonObject.getValue("type").jsonPrimitive.content
        "event: $type\ndata: $json\n\n"
    }

    private fun stub(vararg events: String) {
        wireMock.stubFor(
            post(urlEqualTo("/v1/messages")).willReturn(
                aResponse().withHeader("Content-Type", "text/event-stream").withBody(sse(*events)),
            ),
        )
    }

    private fun messageStart(usage: String = """{"input_tokens":10,"output_tokens":1}""") =
        """{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":$usage}}"""

    private fun textBlock(text: String) = listOf(
        """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}""",
        """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"$text"}}""",
        """{"type":"content_block_stop","index":0}""",
    )

    private fun finish(stopReason: String, outputTokens: Int = 5) = listOf(
        """{"type":"message_delta","delta":{"stop_reason":"$stopReason","stop_sequence":null},"usage":{"output_tokens":$outputTokens}}""",
        """{"type":"message_stop"}""",
    )

    private fun prompt(llm: Llm = sonnet, settings: ToolsPromptSettings? = null): Prompt = buildPrompt {
        this.llm = llm
        this.settings = settings
        system { text("You are a game master.", cacheStrategy = PromptCacheStrategy.PIPELINE) }
        user { text("Hello") }
    }

    private class Outcome(val events: List<PromptExecutionEvent>, val error: Throwable?)

    private fun run(prompt: Prompt, executor: AnthropicPromptExecutor = executor()): Outcome = runBlocking {
        val events = mutableListOf<PromptExecutionEvent>()
        var error: Throwable? = null
        executor.execute(prompt).catch { error = it }.collect { events += it }
        Outcome(events, error)
    }

    private fun requestBody(): JsonObject {
        val requests = wireMock.allServeEvents
        assertEquals(1, requests.size, "exactly one request")
        return Json.parseToJsonElement(requests.single().request.bodyAsString).jsonObject
    }

    // ---- contract of the executor ----

    @Test
    fun `serves the anthropic provider only`() {
        val executor = executor()
        assertTrue(executor.supports(sonnet))
        assertTrue(!executor.supports(Llm(provider = "azure", family = "gpt", name = "x", version = "1", stable = true, model = "gpt")))
    }

    // ---- successful streams ----

    @Test
    fun `plain answer streams text and completes, request carries model, limit, thinking, system and cache mark`() {
        stub(messageStart(), *textBlock("Hi there").toTypedArray(), *finish("end_turn").toTypedArray())

        val outcome = run(prompt())

        assertNull(outcome.error)
        assertEquals(listOf(MessagePart.Text("Hi there")), outcome.events.filterIsInstance<PartReceived>().map { it.part })
        assertIs<PromptStarted>(outcome.events.first())
        assertEquals(Completed(finishReason = "end_turn"), outcome.events.last())

        val body = requestBody()
        assertEquals("claude-sonnet-5-5", body.getValue("model").jsonPrimitive.content)
        assertEquals("true", body.getValue("stream").jsonPrimitive.content)
        assertEquals("128000", body.getValue("max_tokens").jsonPrimitive.content)
        // Sonnet minimum: between_tools, no effort, no tools, no tool_choice.
        assertEquals("between_tools", body.getValue("thinking").jsonObject.getValue("type").jsonPrimitive.content)
        assertNull(body["output_config"])
        assertNull(body["tools"])
        assertNull(body["tool_choice"])
        val system = body.getValue("system").jsonArray.single().jsonObject
        assertEquals("You are a game master.", system.getValue("text").jsonPrimitive.content)
        assertEquals("ephemeral", system.getValue("cache_control").jsonObject.getValue("type").jsonPrimitive.content)
    }

    @Test
    fun `tool_use answer becomes a tool call with the CallId, tools go into the request`() {
        stub(
            messageStart(),
            """{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_42","name":"ping","input":{}}}""",
            """{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}""",
            """{"type":"content_block_stop","index":0}""",
            *finish("tool_use").toTypedArray(),
        )
        val settings = ToolsPromptSettings(tools = buildInstruments { from(PingTool()) }, thinkingLevel = ThinkingLevel.HIGH, maxOutputTokens = 4096)

        val outcome = run(prompt(settings = settings))

        assertNull(outcome.error)
        val call = outcome.events.filterIsInstance<PartReceived>().map { it.part }.single() as MessagePart.ToolCall
        assertEquals("ping", call.name)
        assertEquals(CallId("toolu_42"), call.callId)
        assertEquals(Completed(finishReason = "tool_use"), outcome.events.last())

        val body = requestBody()
        assertEquals("4096", body.getValue("max_tokens").jsonPrimitive.content)
        assertEquals("ping", body.getValue("tools").jsonArray.single().jsonObject.getValue("name").jsonPrimitive.content)
        assertEquals("adaptive", body.getValue("thinking").jsonObject.getValue("type").jsonPrimitive.content)
        assertEquals("high", body.getValue("output_config").jsonObject.getValue("effort").jsonPrimitive.content)
        assertNull(body["tool_choice"])
    }

    @Test
    fun `thinking form follows the level and the model table, the budget is not consulted`() {
        stub(messageStart(), *textBlock("ok").toTypedArray(), *finish("end_turn").toTypedArray())
        // Haiku has no between_tools: its minimum is adaptive + low.
        run(prompt(haiku))
        val body = requestBody()
        assertEquals("adaptive", body.getValue("thinking").jsonObject.getValue("type").jsonPrimitive.content)
        assertEquals("low", body.getValue("output_config").jsonObject.getValue("effort").jsonPrimitive.content)
    }

    @Test
    fun `no level sends temperature, top_p, top_k, a budget or a forced tool_choice`() {
        for (level in ThinkingLevel.entries) {
            val settings = ToolsPromptSettings(tools = buildInstruments { from(PingTool()) }, thinkingLevel = level, thinkingBudget = 1234)
            val body = requestJson(prompt(settings = settings))
            for (forbidden in listOf("temperature", "top_p", "top_k", "tool_choice")) assertNull(body[forbidden], "$forbidden at $level")
            assertNull(body.getValue("thinking").jsonObject["budget_tokens"], "budget_tokens at $level")
        }
    }

    /** The request of [prompt] as the wire JSON the SDK would send. */
    private fun requestJson(prompt: Prompt): JsonObject {
        wireMock.resetRequests()
        stub(messageStart(), *textBlock("ok").toTypedArray(), *finish("end_turn").toTypedArray())
        run(prompt)
        return requestBody()
    }

    @Test
    fun `a request refused by the mapper never reaches the network and carries no PromptStarted`() {
        val pdf = buildPrompt {
            llm = sonnet
            user { text("read"); fileUri("gs://bucket/book.pdf", mimeType = "application/pdf") }
        }
        val outcome = run(pdf)
        assertIs<InvalidRequestException>(outcome.error)
        assertEquals(0, wireMock.allServeEvents.size)
        assertTrue(outcome.events.none { it is PromptStarted })
        assertIs<Failed>(outcome.events.last())
    }

    @Test
    fun `thinking parameters are the same for the same level on every request of a run`() {
        val settings = ToolsPromptSettings(tools = emptyList(), thinkingLevel = ThinkingLevel.MEDIUM, thinkingBudget = 1234)
        val first = prompt(settings = settings).toRequest()
        val second = prompt(settings = settings).toRequest()
        assertEquals(first.thinking(), second.thinking())
        assertEquals(first.outputConfig(), second.outputConfig())
        assertEquals("medium", requestEffort(first))
    }

    // ---- thought visibility and stream silence ----

    private fun thinkingDisplay(): String? =
        requestBody().getValue("thinking").jsonObject["display"]?.jsonPrimitive?.content

    @Test
    fun `includeThoughts asks the API to stream the thought, no flag leaves it hidden`() {
        stub(messageStart(), *textBlock("ok").toTypedArray(), *finish("end_turn").toTypedArray())
        run(prompt(haiku, ToolsPromptSettings(tools = emptyList(), includeThoughts = true)))
        assertEquals("summarized", thinkingDisplay())

        wireMock.resetRequests()
        run(prompt(haiku, ToolsPromptSettings(tools = emptyList(), includeThoughts = false)))
        assertNull(thinkingDisplay())
    }

    /** One stream delivered in two chunks with a gap (> the idle timeout used below) between the first and the second. */
    private fun stubSilentStream() {
        val body = sse(messageStart(), *textBlock("ok").toTypedArray(), *finish("end_turn").toTypedArray())
        wireMock.stubFor(
            post(urlEqualTo("/v1/messages")).willReturn(
                aResponse().withHeader("Content-Type", "text/event-stream").withBody(body).withChunkedDribbleDelay(2, 1600),
            ),
        )
    }

    @Test
    fun `a hidden thought may keep the stream silent past the idle timeout`() {
        stubSilentStream()
        val outcome = run(
            prompt(haiku, ToolsPromptSettings(tools = emptyList(), includeThoughts = false)),
            executor(idleTimeout = 300.milliseconds),
        )
        assertNull(outcome.error)
        assertTrue(outcome.events.any { it is Completed })
    }

    @Test
    fun `a visible thought that goes silent past the idle timeout is a NETWORK_TIMEOUT`() {
        stubSilentStream()
        val outcome = run(
            prompt(haiku, ToolsPromptSettings(tools = emptyList(), includeThoughts = true)),
            executor(idleTimeout = 300.milliseconds),
        )
        val error = assertIs<TransientException>(outcome.error)
        assertEquals(TransientReason.NETWORK_TIMEOUT, error.reason)
    }

    @Test
    fun `cancelling the caller mid-stream is a cancellation, no Failed event and nothing to retry`() = runBlocking {
        wireMock.stubFor(
            post(urlEqualTo("/v1/messages")).willReturn(
                aResponse().withHeader("Content-Type", "text/event-stream")
                    .withBody(sse(messageStart(), *textBlock("ok").toTypedArray(), *finish("end_turn").toTypedArray()))
                    .withChunkedDribbleDelay(2, 20_000),
            ),
        )
        val events = CopyOnWriteArrayList<PromptExecutionEvent>()
        val caught = AtomicReference<Throwable?>()
        val job = launch(Dispatchers.Default) {
            try {
                executor().execute(prompt(haiku, ToolsPromptSettings(tools = emptyList(), includeThoughts = true))).collect { events += it }
            } catch (t: Throwable) {
                caught.set(t)
                throw t
            }
        }
        delay(1_000)
        job.cancelAndJoin()
        assertIs<CancellationException>(assertNotNull(caught.get()))
        assertTrue(events.none { it is Failed })
    }

    private fun requestEffort(request: MessageCreateParams): String =
        request.outputConfig().get().effort().get().asString()

    @Test
    fun `workspace id goes into the header with an api key`() {
        stub(messageStart(), *textBlock("ok").toTypedArray(), *finish("end_turn").toTypedArray())

        run(prompt(), executor(AnthropicAuth.ApiKey("test-key", workspaceId = "wrkspc_test")))

        val request = wireMock.allServeEvents.single().request
        assertEquals("wrkspc_test", request.getHeader(WORKSPACE_HEADER))
        assertEquals("test-key", request.getHeader("x-api-key"))
    }

    @Test
    fun `no workspace header without a workspace id`() {
        stub(messageStart(), *textBlock("ok").toTypedArray(), *finish("end_turn").toTypedArray())

        run(prompt())

        assertNull(wireMock.allServeEvents.single().request.getHeader(WORKSPACE_HEADER))
    }

    // ---- verdicts on the finished response ----

    @Test
    fun `empty answer is a transient EMPTY_RESPONSE, Failed is emitted after the usage`() {
        stub(messageStart("""{"input_tokens":2000,"output_tokens":0}"""), *finish("end_turn", outputTokens = 0).toTypedArray())

        val outcome = run(prompt())

        val error = assertIs<TransientException>(outcome.error)
        assertEquals(TransientReason.EMPTY_RESPONSE, error.reason)
        assertTrue(outcome.events.any { it is PromptExecutionEvent.UsageDelta }, "the paid input must reach the statistics")
        assertNull(outcome.events.firstOrNull { it is Completed })
        assertIs<Failed>(outcome.events.last())
    }

    @Test
    fun `a thought alone is not an empty answer`() {
        stub(
            messageStart(),
            """{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}""",
            """{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}""",
            """{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}""",
            """{"type":"content_block_stop","index":0}""",
            *finish("end_turn").toTypedArray(),
        )

        val outcome = run(prompt())

        assertNull(outcome.error)
        assertIs<MessagePart.Thought>(outcome.events.filterIsInstance<PartReceived>().single().part)
    }

    @Test
    fun `max_tokens is a MaxTokensException even when nothing was produced`() {
        stub(messageStart(), *finish("max_tokens", outputTokens = 128000).toTypedArray())

        val outcome = run(prompt())

        assertIs<MaxTokensException>(outcome.error)
        assertIs<Failed>(outcome.events.last())
    }

    @Test
    fun `refusal is a policy violation that keeps the raw stop reason`() {
        stub(messageStart(), *textBlock("I can't help").toTypedArray(), *finish("refusal").toTypedArray())

        val outcome = run(prompt())

        val error = assertIs<PolicyViolationException>(outcome.error)
        assertEquals("refusal", error.finishReason)
        assertEquals("refusal", assertIs<Failed>(outcome.events.last()).finishReason)
    }

    @Test
    fun `an unexpected stop reason is loud and not accepted as a turn`() {
        stub(messageStart(), *textBlock("partial").toTypedArray(), *finish("pause_turn").toTypedArray())

        assertIs<UnknownProviderException>(run(prompt()).error)
    }

    // ---- transport failures ----

    @Test
    fun `529 overloaded is a transient server error and Failed is emitted`() {
        wireMock.stubFor(
            post(urlEqualTo("/v1/messages")).willReturn(
                aResponse().withStatus(529).withHeader("Content-Type", "application/json")
                    .withBody("""{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"""),
            ),
        )

        val outcome = run(prompt())

        val error = assertIs<TransientException>(outcome.error)
        assertEquals(TransientReason.SERVER_ERROR, error.reason)
        // PromptStarted precedes the request; the provider never answered, so no StreamResponded.
        assertEquals(listOf("PromptStarted", "Failed"), outcome.events.map { it::class.simpleName })
    }

    @Test
    fun `an invalid request is mapped before any retry decision, 400 is not transient`() {
        wireMock.stubFor(
            post(urlEqualTo("/v1/messages")).willReturn(
                aResponse().withStatus(400).withHeader("Content-Type", "application/json")
                    .withBody("""{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}"""),
            ),
        )

        val error = assertNotNull(run(prompt()).error)
        assertTrue(error !is TransientException, "400 must not be retried: $error")
    }

    // ---- stop-reason table, no network ----

    @Test
    fun `stop reason table`() {
        val p = prompt()
        assertNull(judgeStopReason(p, "end_turn", producedContent = true))
        assertNull(judgeStopReason(p, "tool_use", producedContent = true))
        assertNull(judgeStopReason(p, "stop_sequence", producedContent = true))
        assertEquals(TransientReason.EMPTY_RESPONSE, assertIs<TransientException>(judgeStopReason(p, "tool_use", producedContent = false)).reason)
        assertEquals(TransientReason.INCOMPLETE_RESPONSE, assertIs<TransientException>(judgeStopReason(p, null, producedContent = true)).reason)
        assertIs<MaxTokensException>(judgeStopReason(p, "max_tokens", producedContent = true))
        assertIs<PolicyViolationException>(judgeStopReason(p, "refusal", producedContent = false))
        assertIs<UnknownProviderException>(judgeStopReason(p, "pause_turn", producedContent = true))
    }
}
