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
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.EmbeddingPromptSettings
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.TextPromptPart
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * Integration tests for [AzureOpenAiClient]'s embedding path against a [WireMockServer] standing
 * in for Azure OpenAI's Embeddings API. Mirrors [AzureOpenAiClientTest]'s WireMock harness (the
 * SDK stays in default, non-Azure request mode, POSTing to `/embeddings`). Exercises OUR request
 * serialization (model/input/dimensions) and response → event-stream mapping (embedding vector +
 * input-token usage), not Azure's URL scheme.
 */
class AzureOpenAiClientEmbeddingTest {

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

    private fun embeddingPrompt(text: String, dimensions: Int = 768): Prompt = Prompt(
        llm = Llm("azure", "embedding", "Text Embedding 3 Large", "3", true, "text-embedding-3-large"),
        settings = EmbeddingPromptSettings(outputDimensionality = dimensions),
        content = listOf(PromptContent(PromptContentType.USER, listOf(TextPromptPart(text)))),
    )

    @Test
    fun `embedding path serializes model + input + dimensions and maps vector and usage`() = runBlocking {
        wireMock.stubFor(
            post(urlEqualTo("/embeddings")).willReturn(
                aResponse().withHeader("Content-Type", "application/json").withBody(
                    """
                    {
                      "object": "list",
                      "data": [
                        { "object": "embedding", "index": 0, "embedding": [0.1, -0.2, 0.3] }
                      ],
                      "model": "text-embedding-3-large",
                      "usage": { "prompt_tokens": 7, "total_tokens": 7 }
                    }
                    """.trimIndent()
                )
            )
        )

        val client = AzureOpenAiClient(sdkClient, DirectGptStrategy(sdkClient))
        val events = client.execute(embeddingPrompt("dark tense dungeon ambience")).toList()

        // Response parsing: the embedding vector arrives as a single Embedding part.
        val embedding = events.filterIsInstance<PromptExecutionEvent.PartReceived>()
            .map { it.part }
            .filterIsInstance<MessagePart.Embedding>()
            .single()
        assertContentEquals(floatArrayOf(0.1f, -0.2f, 0.3f), embedding.values)
        assertEquals(3, embedding.dimension)

        // Embeddings bill input tokens only — usage carries prompt_tokens as input.total.
        val usage = events.filterIsInstance<PromptExecutionEvent.UsageDelta>().single()
        assertEquals(7L, usage.input.total)
        assertEquals(0L, usage.output.total)

        events.filterIsInstance<PromptExecutionEvent.Completed>().single()

        // Request serialization: input text + the requested reduced dimensionality on the wire.
        wireMock.verify(
            postRequestedFor(urlEqualTo("/embeddings"))
                .withRequestBody(containing("dark tense dungeon ambience"))
                .withRequestBody(containing("\"dimensions\":768"))
                .withRequestBody(containing("\"model\":\"text-embedding-3-large\""))
        )
    }
}
