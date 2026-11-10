package org.nobirds.murmuration.openai

import com.github.tomakehurst.wiremock.WireMockServer
import com.github.tomakehurst.wiremock.client.WireMock.aResponse
import com.github.tomakehurst.wiremock.client.WireMock.post
import com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo
import com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig
import com.openai.client.OpenAIClient
import com.openai.client.okhttp.OpenAIOkHttpClient
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Failed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds

/**
 * OpenAI side: event-order contract of [AzureOpenAiClient] with
 * [DirectGptStrategy] and [StreamingGptStrategy] (with `response.created` and without it — the
 * fallback path): exactly one `PromptStarted`, first; `StreamResponded` once between it and the first
 * part.
 *
 * RED on HEAD: [direct request that fails before any response] — `PromptStarted` is emitted only
 * after the response arrives (`GptResponseStrategy.kt:72`), so a request that reached the wire and
 * failed has no `PromptStarted` at all.
 */
class OpenAiEventOrderContractTest {

    private lateinit var wireMock: WireMockServer
    private lateinit var sdkClient: OpenAIClient

    @BeforeEach
    fun setUp() {
        wireMock = WireMockServer(wireMockConfig().dynamicPort())
        wireMock.start()
        sdkClient = OpenAIOkHttpClient.builder().baseUrl(wireMock.baseUrl()).apiKey("test").maxRetries(0).build()
    }

    @AfterEach
    fun tearDown() = wireMock.stop()

    private fun testPrompt(): Prompt = buildPrompt {
        llm = Llm("azure", "gpt", "Test", "1", true, "gpt-5.4-nano")
        user { text("hi") }
    }

    private fun envelope(status: String, model: String = "gpt-5.4-nano"): String =
        """{"id":"resp_test","object":"response","created_at":1700000000,"model":"$model","status":"$status","output":[]}"""

    private fun dataLine(json: String): String = "data: $json\n\n"

    private fun stubSse(withCreated: Boolean, createdModel: String = "gpt-5.4-nano") {
        val sse = buildString {
            if (withCreated) append(dataLine("""{"type":"response.created","sequence_number":0,"response":${envelope("in_progress", createdModel)}}"""))
            append(dataLine("""{"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"hello","logprobs":[]}"""))
            append(dataLine("""{"type":"response.completed","sequence_number":2,"response":${envelope("completed")}}"""))
            append("data: [DONE]\n\n")
        }
        wireMock.stubFor(post(urlEqualTo("/responses")).willReturn(aResponse().withHeader("Content-Type", "text/event-stream").withBody(sse)))
    }

    private fun stubDirect() = wireMock.stubFor(
        post(urlEqualTo("/responses")).willReturn(
            aResponse().withHeader("Content-Type", "application/json").withBody(
                """{"id":"resp_test","object":"response","created_at":1700000000,"model":"gpt-5.4-nano","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}]}""",
            ),
        ),
    )

    private fun run(strategy: GptResponseStrategy): List<PromptExecutionEvent> = runBlocking {
        val events = mutableListOf<PromptExecutionEvent>()
        try {
            AzureOpenAiClient(sdkClient, strategy).execute(testPrompt()).toList(events)
        } catch (_: Throwable) {
            // The terminal failure is visible through the Failed event.
        }
        events
    }

    private fun streaming() = StreamingGptStrategy(sdkClient, 30.seconds, 120.seconds)

    @Test
    fun `direct success - one PromptStarted first, parts follow`() {
        stubDirect()
        val events = run(DirectGptStrategy(sdkClient))

        assertPromptStartedFirstAndOnce(events)
        assertTrue(events.any { it is PartReceived } && events.last() is Completed, "events=${events.names()}")
    }

    @Test
    fun `direct request that fails before any response - PromptStarted still precedes Failed`() {
        wireMock.stubFor(post(urlEqualTo("/responses")).willReturn(aResponse().withStatus(500).withBody("{}")))
        val events = run(DirectGptStrategy(sdkClient))

        assertPromptStartedFirstAndOnce(events)
        assertTrue(events.last() is Failed, "events=${events.names()}")
    }

    @Test
    fun `stream with response created - one PromptStarted first`() {
        stubSse(withCreated = true)
        val events = run(streaming())

        assertPromptStartedFirstAndOnce(events)
        assertTrue(events.any { it is PartReceived } && events.last() is Completed, "events=${events.names()}")
    }

    @Test
    fun `stream without response created (fallback path) - still one PromptStarted first`() {
        stubSse(withCreated = false)
        val events = run(streaming())

        assertPromptStartedFirstAndOnce(events)
        assertTrue(events.last() is Completed, "events=${events.names()}")
    }

    @Test
    fun `direct success - StreamResponded once between PromptStarted and the first part`() {
        stubDirect()
        assertStreamRespondedOnce(run(DirectGptStrategy(sdkClient)))
    }

    @Test
    fun `stream with response created - StreamResponded once between PromptStarted and the first part`() {
        stubSse(withCreated = true)
        assertStreamRespondedOnce(run(streaming()))
    }

    @Test
    fun `stream without response created - StreamResponded once before completion`() {
        stubSse(withCreated = false)
        assertStreamRespondedOnce(run(streaming()))
    }

    // --- F3 (best-effort served model): documents what `servedModelName` is on this path ---

    private fun servedModelOf(events: List<PromptExecutionEvent>): String =
        events.filterIsInstance<PromptExecutionEvent.StreamResponded>().single().servedModelName

    @Test
    fun `stream - served model comes from response created when it is the first event`() {
        stubSse(withCreated = true, createdModel = "gpt-5.4-nano-served")
        assertEquals("gpt-5.4-nano-served", servedModelOf(run(streaming())))
    }

    @Test
    fun `stream without response created - served model falls back to the requested one`() {
        stubSse(withCreated = false)
        assertEquals("gpt-5.4-nano", servedModelOf(run(streaming())))
    }
}
