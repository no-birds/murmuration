package org.nobirds.murmuration.openai

import com.github.tomakehurst.wiremock.WireMockServer
import com.github.tomakehurst.wiremock.client.WireMock.aResponse
import com.github.tomakehurst.wiremock.client.WireMock.post
import com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo
import com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig
import com.openai.client.okhttp.OpenAIOkHttpClient
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.agent.execute
import org.nobirds.murmuration.agent.executePipeline
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.prompt.withReplaceableSectionPruning
import org.nobirds.murmuration.tooling.Instrument
import org.nobirds.murmuration.tooling.ToolCallingInput
import org.nobirds.murmuration.tooling.toolCallingAgent
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Wire-level regression for reasoning replay: drives the REAL [org.nobirds.murmuration.tooling.ToolCallingAgent]
 * loop against [AzureOpenAiClient] (wrapped in the same section-pruning decorator production
 * uses) over WireMock, and asserts the SECOND HTTP request body physically carries the
 * `reasoning` input item with the original `encrypted_content` — the one thing in-process unit
 * tests can't see (telemetry hides part signatures, so a silent drop is otherwise invisible).
 */
class ReplayHttpProbeTest {

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

    @Test
    fun `second HTTP request body carries the reasoning item with the original encrypted content`() = runBlocking {
        val rt1 = """
        {
          "id": "resp_1", "object": "response", "created_at": 1700000000,
          "model": "gpt-5.4-nano", "status": "completed",
          "output": [
            {"type": "reasoning", "id": "rs_probe", "summary": [], "encrypted_content": "PROBE_ENC_BLOB"},
            {"type": "function_call", "id": "fc_1", "call_id": "call_x", "name": "ping", "arguments": "{}", "status": "completed"}
          ],
          "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 3}, "total_tokens": 15}
        }
        """.trimIndent()
        val rt2 = """
        {
          "id": "resp_2", "object": "response", "created_at": 1700000001,
          "model": "gpt-5.4-nano", "status": "completed",
          "output": [
            {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "done", "annotations": []}]}
          ],
          "usage": {"input_tokens": 20, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 2, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 22}
        }
        """.trimIndent()
        wireMock.stubFor(
            post(urlEqualTo("/responses")).inScenario("replay").whenScenarioStateIs("Started")
                .willReturn(aResponse().withHeader("Content-Type", "application/json").withBody(rt1))
                .willSetStateTo("second")
        )
        wireMock.stubFor(
            post(urlEqualTo("/responses")).inScenario("replay").whenScenarioStateIs("second")
                .willReturn(aResponse().withHeader("Content-Type", "application/json").withBody(rt2))
        )

        val sdk = OpenAIOkHttpClient.builder().baseUrl(wireMock.baseUrl()).apiKey("test").maxRetries(0).build()
        // Same decorator chain production uses (Application.kt wraps the router in pruning).
        val executor = AzureOpenAiClient(sdk, DirectGptStrategy(sdk)).withReplaceableSectionPruning(enabled = true)
        val llm = Llm("azure", "gpt", "Test", "1", true, "gpt-5.4-nano")

        executePipeline {
            val agent = toolCallingAgent<Unit, PingTool>(
                llm = llm, executor = executor, name = "ProbeAgent", maxTurns = 2,
            ) { _, _, _ -> user { text("probe") } }
            execute(agent, ToolCallingInput(Unit, PingTool()))
        }

        val bodies = wireMock.allServeEvents.sortedBy { it.request.loggedDate }.map { it.request.bodyAsString }
        assertEquals(2, bodies.size, "agent must issue exactly two requests")
        assertTrue("PROBE_ENC_BLOB" !in bodies[0], "first request cannot contain a not-yet-returned reasoning blob")
        // Request-shape pins: readable summaries requested, parallel tool calls explicit.
        assertTrue(""""summary":"auto"""" in bodies[0], "requests must ask for reasoning summaries (summary=auto)")
        assertTrue(""""parallel_tool_calls":true""" in bodies[0], "parallel_tool_calls must be pinned explicitly")

        val second = bodies[1]
        assertTrue(""""type":"reasoning"""" in second, "second request must carry a reasoning input item; body: ${second.take(2000)}")
        assertTrue(""""encrypted_content":"PROBE_ENC_BLOB"""" in second, "the original encrypted content must be replayed verbatim")
        assertTrue(""""id":"rs_probe"""" in second, "the original reasoning item id must be replayed")
        // Atomic pairing on the wire: the reasoning item precedes its function call.
        val reasoningIdx = second.indexOf(""""type":"reasoning"""")
        val callIdx = second.indexOf(""""type":"function_call"""")
        assertTrue(callIdx > reasoningIdx, "function_call must follow the reasoning item")
    }
}
