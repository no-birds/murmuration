package org.nobirds.murmuration.openai

import com.github.tomakehurst.wiremock.WireMockServer
import com.github.tomakehurst.wiremock.client.WireMock.aResponse
import com.github.tomakehurst.wiremock.client.WireMock.containing
import com.github.tomakehurst.wiremock.client.WireMock.post
import com.github.tomakehurst.wiremock.client.WireMock.postRequestedFor
import com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo
import com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig
import com.openai.client.OpenAIClient
import com.openai.client.okhttp.OpenAIOkHttpClient
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.agent.AgentExecutionIdentity
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.ImagePromptSettings
import org.nobirds.murmuration.prompt.MaxTokensException
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.image
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import java.util.Base64
import java.util.UUID
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds

/**
 * Integration tests for [AzureOpenAiClient] against a [WireMockServer] standing in for Azure
 * OpenAI. The [OpenAIClient] is built directly via [OpenAIOkHttpClient] with a plain `baseUrl`
 * (NOT [com.openai.azure.AzureUrlPathMode]) — the SDK stays in its default, non-Azure request
 * mode, which is exactly what's needed here: these tests exercise OUR event-stream mapping
 * ([DirectGptStrategy] / [StreamingGptStrategy] / [Response.mapOutcome]), not Azure's URL scheme.
 */
class AzureOpenAiClientTest {

    private lateinit var wireMock: WireMockServer
    private lateinit var sdkClient: OpenAIClient

    @BeforeEach
    fun setUp() {
        wireMock = WireMockServer(wireMockConfig().dynamicPort())
        wireMock.start()
        sdkClient = OpenAIOkHttpClient.builder()
            .baseUrl(wireMock.baseUrl())
            .apiKey("test")
            .maxRetries(0)
            .build()
    }

    @AfterEach
    fun tearDown() {
        wireMock.stop()
    }

    private fun testPrompt(): Prompt = buildPrompt {
        llm = Llm("azure", "gpt", "Test", "1", true, "gpt-5.4-nano")
        user { text("What's the weather in Paris?") }
    }

    @Test
    fun `direct strategy emits reasoning, text, tool call, usage and completion`() = runBlocking {
        wireMock.stubFor(
            post(urlEqualTo("/responses")).willReturn(
                aResponse().withHeader("Content-Type", "application/json").withBody(
                    """
                    {
                      "id": "resp_test",
                      "object": "response",
                      "created_at": 1700000000,
                      "model": "gpt-5.4-nano",
                      "status": "completed",
                      "output": [
                        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Checking the weather tool."}], "encrypted_content": "ENC"},
                        {
                          "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                          "content": [{"type": "output_text", "text": "Let me check that.", "annotations": []}]
                        },
                        {
                          "type": "function_call", "id": "fc_1", "call_id": "call_abc",
                          "name": "get_weather", "arguments": "{\"city\":\"Paris\"}", "status": "completed"
                        }
                      ],
                      "usage": {
                        "input_tokens": 50,
                        "input_tokens_details": {"cached_tokens": 10},
                        "output_tokens": 20,
                        "output_tokens_details": {"reasoning_tokens": 5},
                        "total_tokens": 70
                      }
                    }
                    """.trimIndent()
                )
            )
        )

        val client = AzureOpenAiClient(sdkClient, DirectGptStrategy(sdkClient))
        val events = client.execute(testPrompt()).toList()
        val parts = events.filterIsInstance<PromptExecutionEvent.PartReceived>().map { it.part }

        assertTrue(events.any { it is PromptExecutionEvent.PromptStarted })

        assertEquals("Let me check that.", parts.filterIsInstance<MessagePart.Text>().single().text)

        // The reasoning item's signature rides on the FOLLOWING tool-call part (the channel the
        // core agent loop round-trips verbatim into the history), not on a separate Thought part.
        val toolCall = parts.filterIsInstance<MessagePart.ToolCall>().single()
        assertEquals("get_weather", toolCall.name)
        assertEquals("Paris", (toolCall.args["city"] as JsonPrimitive).content)
        val decoded = decodeReasoningSignature(toolCall.thoughtSignature)!!
        assertEquals("rs_1", decoded.id)
        assertEquals("ENC", decoded.enc)

        // The readable summary comes out as a marker-signed Thought: text for telemetry, but a
        // signature WITHOUT encrypted content — never replayed, never mistaken for assistant text.
        val thought = parts.filterIsInstance<MessagePart.Thought>().single()
        assertEquals("Checking the weather tool.", thought.text)
        val marker = decodeReasoningSignature(thought.signature)!!
        assertEquals("rs_1", marker.id)
        assertEquals(null, marker.enc)

        val usage = events.filterIsInstance<PromptExecutionEvent.UsageDelta>().single()
        assertEquals(50L, usage.input.total)
        assertEquals(10L, usage.input.cached)
        assertEquals(15L, usage.output.total) // 20 output - 5 reasoning
        assertEquals(5L, usage.output.thinking)

        assertEquals("completed", events.filterIsInstance<PromptExecutionEvent.Completed>().single().finishReason)
    }

    @Test
    fun `streaming strategy emits text deltas, a reasoning-signed tool call, and final usage`() = runBlocking {
        val sse = buildString {
            append(dataLine("""{"type":"response.created","sequence_number":0,"response":${responseEnvelope("in_progress", withUsage = false)}}"""))
            append(dataLine("""{"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"Sunny in ","logprobs":[]}"""))
            append(dataLine("""{"type":"response.output_text.delta","sequence_number":2,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"Paris","logprobs":[]}"""))
            append(
                dataLine(
                    """{"type":"response.reasoning_summary_text.done","sequence_number":3,"item_id":"rs_s1","output_index":1,"summary_index":0,"text":"Streamed reasoning summary."}"""
                )
            )
            append(
                dataLine(
                    """{"type":"response.output_item.done","sequence_number":4,"output_index":1,"item":{"type":"reasoning","id":"rs_s1","summary":[{"type":"summary_text","text":"Streamed reasoning summary."}],"encrypted_content":"ENC_STREAM"}}"""
                )
            )
            append(
                dataLine(
                    """{"type":"response.output_item.done","sequence_number":5,"output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call_abc","name":"get_weather","arguments":"{\"city\":\"Paris\"}","status":"completed"}}"""
                )
            )
            append(dataLine("""{"type":"response.completed","sequence_number":6,"response":${responseEnvelope("completed", withUsage = true)}}"""))
            append("data: [DONE]\n\n")
        }

        wireMock.stubFor(
            post(urlEqualTo("/responses")).willReturn(
                aResponse().withHeader("Content-Type", "text/event-stream").withBody(sse)
            )
        )

        val client = AzureOpenAiClient(sdkClient, StreamingGptStrategy(sdkClient, 30.seconds, 120.seconds))
        val events = client.execute(testPrompt()).toList()
        val parts = events.filterIsInstance<PromptExecutionEvent.PartReceived>().map { it.part }

        val text = parts.filterIsInstance<MessagePart.Text>().joinToString("") { it.text }
        assertEquals("Sunny in Paris", text)

        val toolCall = parts.filterIsInstance<MessagePart.ToolCall>().single()
        assertEquals("get_weather", toolCall.name)
        assertEquals("Paris", (toolCall.args["city"] as JsonPrimitive).content)
        // Streamed reasoning item (output_item.done) rides the following tool-call's signature.
        val streamedReasoning = decodeReasoningSignature(toolCall.thoughtSignature)!!
        assertEquals("rs_s1", streamedReasoning.id)
        assertEquals("ENC_STREAM", streamedReasoning.enc)

        // Exactly ONE readable Thought (from reasoning_summary_text.done; the completed item's
        // embedded summary must not double-emit), marker-signed (no encrypted content).
        val thoughts = parts.filterIsInstance<MessagePart.Thought>()
        assertEquals(1, thoughts.size)
        assertEquals("Streamed reasoning summary.", thoughts.single().text)
        assertEquals(null, decodeReasoningSignature(thoughts.single().signature)!!.enc)

        val usageDeltas = events.filterIsInstance<PromptExecutionEvent.UsageDelta>()
        assertEquals(1, usageDeltas.size)
        assertEquals(30L, usageDeltas[0].input.total)
        assertEquals(10L, usageDeltas[0].output.total)

        assertEquals("completed", events.filterIsInstance<PromptExecutionEvent.Completed>().single().finishReason)
    }

    @Test
    fun `overall round-trip timeout raises a retryable NETWORK_TIMEOUT even while events keep arriving`() = runBlocking {
        // A well-formed stream (events keep arriving, so the per-event idle timeout never trips),
        // but a zero overall budget: the hard cap must fire on the first loop turn with a retryable
        // NETWORK_TIMEOUT — this is the class of hang idle alone cannot catch (reasoning deltas reset idle).
        val sse = buildString {
            append(dataLine("""{"type":"response.created","sequence_number":0,"response":${responseEnvelope("in_progress", withUsage = false)}}"""))
            append(dataLine("""{"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"Sunny","logprobs":[]}"""))
            append(dataLine("""{"type":"response.completed","sequence_number":2,"response":${responseEnvelope("completed", withUsage = true)}}"""))
            append("data: [DONE]\n\n")
        }
        wireMock.stubFor(
            post(urlEqualTo("/responses")).willReturn(
                aResponse().withHeader("Content-Type", "text/event-stream").withBody(sse)
            )
        )

        // idle generous (30s), overall zero → deadline already passed at the first loop check.
        val client = AzureOpenAiClient(sdkClient, StreamingGptStrategy(sdkClient, 30.seconds, 0.seconds))
        val error = assertFailsWith<TransientException> {
            client.execute(testPrompt()).collect()
        }
        assertEquals(TransientReason.NETWORK_TIMEOUT, error.reason)
        assertTrue(error.message!!.contains("overall round-trip budget"), "message should name the overall cap: ${error.message}")
    }

    @Test
    fun `incomplete max_output_tokens raises MaxTokensException and emits Failed`() = runBlocking {
        wireMock.stubFor(
            post(urlEqualTo("/responses")).willReturn(
                aResponse().withHeader("Content-Type", "application/json").withBody(
                    """
                    {
                      "id": "resp_test",
                      "object": "response",
                      "created_at": 1700000000,
                      "model": "gpt-5.4-nano",
                      "status": "incomplete",
                      "incomplete_details": {"reason": "max_output_tokens"},
                      "output": [
                        {
                          "type": "message", "id": "msg_1", "role": "assistant", "status": "incomplete",
                          "content": [{"type": "output_text", "text": "Truncated...", "annotations": []}]
                        }
                      ],
                      "usage": {
                        "input_tokens": 5,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 5,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 10
                      }
                    }
                    """.trimIndent()
                )
            )
        )

        val client = AzureOpenAiClient(sdkClient, DirectGptStrategy(sdkClient))
        val events = mutableListOf<PromptExecutionEvent>()
        assertFailsWith<MaxTokensException> {
            client.execute(testPrompt()).collect { events.add(it) }
        }
        assertTrue(events.any { it is PromptExecutionEvent.Failed })
    }

    // Minimal successful Responses-API body — these tests only care about the REQUEST the SDK
    // sent (prompt_cache_key), not about parsing a realistic response.
    private fun minimalCompletedResponseJson(): String = """
        {
          "id": "resp_test",
          "object": "response",
          "created_at": 1700000000,
          "model": "gpt-5.4-nano",
          "status": "completed",
          "output": [],
          "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 2
          }
        }
    """.trimIndent()

    private fun extractPromptCacheKey(body: String): String? =
        Regex(""""prompt_cache_key":"([^"]*)"""").find(body)?.groupValues?.get(1)

    @Test
    fun `prompt_cache_key is forwarded to the request when an AgentExecutionIdentity is in the collector's context`() = runBlocking {
        wireMock.stubFor(
            post(urlEqualTo("/responses")).willReturn(
                aResponse().withHeader("Content-Type", "application/json").withBody(minimalCompletedResponseJson())
            )
        )

        val client = AzureOpenAiClient(sdkClient, DirectGptStrategy(sdkClient))
        withContext(AgentExecutionIdentity(UUID.randomUUID(), "CharacterEquipmentAgent#a1b2[CallToolAgent]")) {
            client.execute(testPrompt()).toList()
        }

        wireMock.verify(
            postRequestedFor(urlEqualTo("/responses"))
                .withRequestBody(containing(""""prompt_cache_key":"CharacterEquipmentAgent#a1b2[CallToolAgent]""""))
        )
    }

    @Test
    fun `no prompt_cache_key field is sent when no AgentExecutionIdentity is in context`() = runBlocking {
        wireMock.stubFor(
            post(urlEqualTo("/responses")).willReturn(
                aResponse().withHeader("Content-Type", "application/json").withBody(minimalCompletedResponseJson())
            )
        )

        val client = AzureOpenAiClient(sdkClient, DirectGptStrategy(sdkClient))
        client.execute(testPrompt()).toList()

        val request = wireMock.findAll(postRequestedFor(urlEqualTo("/responses"))).single()
        assertTrue(!request.bodyAsString.contains("prompt_cache_key"), "unexpected prompt_cache_key in: ${request.bodyAsString}")
    }

    @Test
    fun `prompt_cache_key is stable across an agent's own round-trips and differs across agents`() = runBlocking {
        wireMock.stubFor(
            post(urlEqualTo("/responses")).willReturn(
                aResponse().withHeader("Content-Type", "application/json").withBody(minimalCompletedResponseJson())
            )
        )

        val client = AzureOpenAiClient(sdkClient, DirectGptStrategy(sdkClient))
        val pipelineId = UUID.randomUUID()
        // Both names carry the shape ToolCallingAgent/TextPromptAgent actually produce
        // (`generateAgentName`): base name + per-instance random suffix + `[Type]` tag.
        val agentAName = "CharacterEquipmentAgent#a1b2[CallToolAgent]"
        val agentBName = "CharacterDerivationAgent#c3d4[CallToolAgent]"

        // Same agent invocation, two round-trips (mirrors ToolCallingAgent's loop wrapping every
        // round-trip's collect in the same displayName-derived identity) — same key.
        withContext(AgentExecutionIdentity(pipelineId, agentAName)) {
            client.execute(testPrompt()).toList()
            client.execute(testPrompt()).toList()
        }
        // A different agent (different instance/name) of the same pipeline — a different key.
        withContext(AgentExecutionIdentity(pipelineId, agentBName)) {
            client.execute(testPrompt()).toList()
        }

        val requests = wireMock.findAll(postRequestedFor(urlEqualTo("/responses")))
        assertEquals(3, requests.size)
        val keys = requests.map { extractPromptCacheKey(it.bodyAsString) }
        assertEquals(listOf(agentAName, agentAName, agentBName), keys)
    }

    @Test
    fun `image prompt without input images calls the generations endpoint`() = runBlocking {
        val imageBytes = byteArrayOf(1, 2, 3, 4)
        wireMock.stubFor(
            post(urlEqualTo("/images/generations")).willReturn(
                aResponse().withHeader("Content-Type", "application/json").withBody(imagesResponseJson(imageBytes))
            )
        )

        val client = AzureOpenAiClient(sdkClient, DirectGptStrategy(sdkClient))
        val prompt = buildPrompt {
            llm = Llm("azure", "gpt", "Test", "1", true, "gpt-image-2")
            settings = ImagePromptSettings()
            user { text("A watercolor castle") }
        }
        val events = client.execute(prompt).toList()

        val inline = events.filterIsInstance<PromptExecutionEvent.PartReceived>()
            .map { it.part }.filterIsInstance<MessagePart.InlineData>().single()
        assertContentEquals(imageBytes, inline.data)

        val usage = events.filterIsInstance<PromptExecutionEvent.UsageDelta>().single()
        assertEquals(10L, usage.input.total)
        assertEquals(5L, usage.output.total)

        wireMock.verify(0, postRequestedFor(urlEqualTo("/images/edits")))
    }

    @Test
    fun `image prompt with an inline input image calls the edits endpoint with multipart`() = runBlocking {
        val resultBytes = byteArrayOf(9, 8, 7)
        wireMock.stubFor(
            post(urlEqualTo("/images/edits")).willReturn(
                aResponse().withHeader("Content-Type", "application/json").withBody(imagesResponseJson(resultBytes))
            )
        )

        val client = AzureOpenAiClient(sdkClient, DirectGptStrategy(sdkClient))
        val prompt = buildPrompt {
            llm = Llm("azure", "gpt", "Test", "1", true, "gpt-image-2")
            settings = ImagePromptSettings()
            user {
                text("Redraw this portrait in watercolor")
                image(byteArrayOf(42, 43, 44), "image/png")
            }
        }
        val events = client.execute(prompt).toList()

        val inline = events.filterIsInstance<PromptExecutionEvent.PartReceived>()
            .map { it.part }.filterIsInstance<MessagePart.InlineData>().single()
        assertContentEquals(resultBytes, inline.data)
        assertTrue(events.any { it is PromptExecutionEvent.Completed })

        wireMock.verify(0, postRequestedFor(urlEqualTo("/images/generations")))
        wireMock.verify(
            1,
            postRequestedFor(urlEqualTo("/images/edits"))
                .withHeader("Content-Type", containing("multipart/form-data"))
                .withRequestBody(containing("Redraw this portrait in watercolor")),
        )
    }

    @Test
    fun `image edit request carries images in prompt order and requests high input fidelity`() = runBlocking {
        val resultBytes = byteArrayOf(9, 8, 7)
        wireMock.stubFor(
            post(urlEqualTo("/images/edits")).willReturn(
                aResponse().withHeader("Content-Type", "application/json").withBody(imagesResponseJson(resultBytes))
            )
        )

        val client = AzureOpenAiClient(sdkClient, DirectGptStrategy(sdkClient))
        val prompt = buildPrompt {
            llm = Llm("azure", "gpt", "Test", "1", true, "gpt-image-2")
            settings = ImagePromptSettings()
            user {
                text("Redraw the skeleton, matching the style sample")
                // Order matters: the FIRST image here must be the one the edit API applies
                // fidelity/mask semantics to (see the KDoc on buildEditParams). Two distinct
                // ASCII-ish payloads so their order is verifiable in the raw multipart body.
                image("FIRST-IMAGE-MARKER".toByteArray(), "image/png")
                image("SECOND-IMAGE-MARKER".toByteArray(), "image/png")
            }
        }
        client.execute(prompt).toList()

        val requests = wireMock.findAll(postRequestedFor(urlEqualTo("/images/edits")))
        assertEquals(1, requests.size)
        val body = requests.single().bodyAsString

        val firstIndex = body.indexOf("FIRST-IMAGE-MARKER")
        val secondIndex = body.indexOf("SECOND-IMAGE-MARKER")
        assertTrue(firstIndex >= 0 && secondIndex >= 0, "both image markers must be present in the multipart body")
        assertTrue(firstIndex < secondIndex, "the first prompt image must precede the second in the image[] multipart parts")

        assertTrue(body.contains("input_fidelity"), "edit request must set input_fidelity")
        assertTrue(body.contains("high"), "edit request must request HIGH input fidelity")
    }

    private fun imagesResponseJson(imageBytes: ByteArray): String {
        val b64 = Base64.getEncoder().encodeToString(imageBytes)
        return """
            {
              "created": 1700000000,
              "data": [{"b64_json": "$b64"}],
              "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"image_tokens": 4, "text_tokens": 6},
                "output_tokens": 5,
                "total_tokens": 15
              }
            }
        """.trimIndent()
    }

    // A minimal Response object embedded in the `response.created` / `response.completed` stream
    // events. `created` carries no usage yet; `completed` carries the final usage the mapper reads.
    private fun responseEnvelope(status: String, withUsage: Boolean): String {
        val usage = if (withUsage) {
            ""","usage":{"input_tokens":30,"input_tokens_details":{"cached_tokens":0},"output_tokens":10,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":40}"""
        } else ""
        return """{"id":"resp_test","object":"response","created_at":1700000000,"model":"gpt-5.4-nano","status":"$status","output":[]$usage}"""
    }

    private fun dataLine(json: String): String = "data: $json\n\n"
}
