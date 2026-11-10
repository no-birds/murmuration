package org.nobirds.murmuration.anthropic

import com.github.tomakehurst.wiremock.WireMockServer
import com.github.tomakehurst.wiremock.client.WireMock.aResponse
import com.github.tomakehurst.wiremock.client.WireMock.post
import com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo
import com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Failed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutionEvent.StreamResponded
import org.nobirds.murmuration.prompt.PromptExecutionEvent.UsageDelta
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * Event-order contract of [AnthropicPromptExecutor] (the same contract as `GeminiEventOrderContractTest`): ONE
 * `PromptStarted`, first, on every path that reaches the request; `StreamResponded` once, on the provider's first event
 * (`message_start`), after `PromptStarted` and before the usage and the first part; none when the provider never answered.
 * Canonical shape: serve a response with WireMock, [run] the executor, compare the event names.
 */
class AnthropicEventOrderContractTest {

    private lateinit var wireMock: WireMockServer

    @BeforeEach
    fun setUp() {
        wireMock = WireMockServer(wireMockConfig().dynamicPort())
        wireMock.start()
    }

    @AfterEach
    fun tearDown() = wireMock.stop()

    private val prompt: Prompt = buildPrompt {
        llm = AnthropicLlms.CLAUDE_SONNET_5_5.llm
        user { text("hi") }
    }

    private fun sse(vararg events: String): String = events.joinToString("") { json ->
        "event: ${Regex(""""type":"(\w+)"""").find(json)!!.groupValues[1]}\ndata: $json\n\n"
    }

    private fun serve(body: String) = wireMock.stubFor(
        post(urlEqualTo("/v1/messages")).willReturn(aResponse().withHeader("Content-Type", "text/event-stream").withBody(body)),
    )

    private val messageStart =
        """{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"served-model","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}"""
    private val textBlock = listOf(
        """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}""",
        """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}""",
        """{"type":"content_block_stop","index":0}""",
    )
    private val finish = listOf(
        """{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}""",
        """{"type":"message_stop"}""",
    )

    private fun run(): List<PromptExecutionEvent> = runBlocking {
        val events = mutableListOf<PromptExecutionEvent>()
        val executor = AnthropicPromptExecutor(AnthropicClientFactory.create(AnthropicAuth.ApiKey("k"), "http://localhost:${wireMock.port()}"))
        try {
            executor.execute(prompt).toList(events)
        } catch (_: Throwable) {
            // The terminal failure is visible through the Failed event.
        }
        events
    }

    private fun List<PromptExecutionEvent>.names() = map { it::class.simpleName }

    @Test
    fun `success - PromptStarted first and once, StreamResponded once before the usage and the first part`() {
        serve(sse(messageStart, *textBlock.toTypedArray(), *finish.toTypedArray()))

        val events = run()

        assertEquals(listOf("PromptStarted", "StreamResponded", "UsageDelta", "PartReceived", "UsageDelta", "Completed"), events.names())
        assertEquals(StreamResponded("served-model"), events[1])
        assertEquals("claude-sonnet-5-5", assertIs<PromptStarted>(events.first()).requestedModelName)
        assertIs<PartReceived>(events[3])
        assertIs<Completed>(events.last())
    }

    @Test
    fun `an HTTP error before any answer - PromptStarted then Failed, no StreamResponded`() {
        wireMock.stubFor(post(urlEqualTo("/v1/messages")).willReturn(aResponse().withStatus(529).withBody("""{"type":"error","error":{"type":"overloaded_error","message":"busy"}}""")))

        val events = run()

        assertEquals(listOf("PromptStarted", "Failed"), events.names())
        assertTrue(events.last() is Failed)
    }

    @Test
    fun `a stream that breaks after message_start - one PromptStarted, one StreamResponded, then Failed`() {
        serve(sse(messageStart, *textBlock.toTypedArray()))

        val events = run()

        assertEquals(1, events.count { it is PromptStarted })
        assertEquals(1, events.count { it is StreamResponded })
        assertEquals(0, events.indexOfFirst { it is PromptStarted })
        assertEquals(1, events.indexOfFirst { it is StreamResponded })
        assertTrue(events.none { it is UsageDelta && events.indexOf(it) < 2 })
        assertTrue(events.last() is Failed, "events=${events.names()}")
    }
}
