package org.nobirds.murmuration.openai

import org.nobirds.murmuration.prompt.CallIds
import com.openai.client.OpenAIClient
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.schema.SchemaInfo
import org.nobirds.murmuration.prompt.EmbeddingPromptSettings
import org.nobirds.murmuration.prompt.ImageAspectRatio
import org.nobirds.murmuration.prompt.ImagePromptSettings
import org.nobirds.murmuration.prompt.ImageResolution
import org.nobirds.murmuration.prompt.ImageSize
import org.nobirds.murmuration.prompt.InstrumentDeclaration
import org.nobirds.murmuration.prompt.InstrumentParameter
import org.nobirds.murmuration.prompt.InstrumentsDeclaration
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.ThinkingLevel
import org.nobirds.murmuration.prompt.ToolsPromptSettings
import org.nobirds.murmuration.prompt.TypedPromptSettings
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.ToolCallPromptPart
import org.nobirds.murmuration.prompt.ToolResultPromptPart
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.image
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import kotlinx.serialization.json.JsonPrimitive
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.io.File
import javax.imageio.ImageIO
import kotlin.reflect.KType
import kotlin.reflect.typeOf
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds

/**
 * Manual, real-endpoint tests for [AzureOpenAiClient] — mirror of gemini's
 * `GeminiImageGenerationTest` disabled-manual-run style. Needs a live Azure OpenAI resource
 * plus `az login` (picked up by [DefaultAzureCredentialBuilder][com.azure.identity.DefaultAzureCredentialBuilder]
 * via [azureBearerTokenSupplier]) and every run bills the deployment. Run explicitly with
 * approval only — never from CI.
 *
 * The whole point of a manual run is to EYEBALL what the model generated, so every test
 * dumps the full event trace (text, tool calls with args, usage, finish reason) to stdout
 * and, when `AZURE_TEST_REPORT_FILE` is set, appends it to that file for post-run review.
 */
@Disabled("Manual: requires real Azure OpenAI endpoint + az login; costs money; run only with explicit approval")
class AzureRealModelIntegrationTest {

    private val endpoint = System.getenv("AZURE_OPENAI_ENDPOINT") ?: ""
    // gpt-5-mini: default deployment for these manual tests (gpt-5.4-nano was also deployed
    // 2026-07-04, GlobalStandard 500K TPM, but isn't the default here).
    private val deployment = System.getenv("AZURE_OPENAI_TEST_DEPLOYMENT") ?: "gpt-5-mini"
    // gpt-image-2: default deployment name matches the model name.
    private val imageDeployment = System.getenv("AZURE_OPENAI_TEST_IMAGE_DEPLOYMENT") ?: "gpt-image-2"
    // text-embedding-3-large: default embedding deployment.
    private val embeddingDeployment = System.getenv("AZURE_OPENAI_TEST_EMBEDDING_DEPLOYMENT") ?: "text-embedding-3-large"

    private fun sdkClient(): OpenAIClient = azureOpenAiClient(endpoint, azureBearerTokenSupplier())

    private fun testLlm(): Llm = Llm("azure", "gpt", "Test", "1", true, deployment)

    /** Dumps the generated content of one test run — stdout + optional report file. */
    private fun report(test: String, events: List<PromptExecutionEvent>) {
        val lines = buildList {
            add("=== $test [deployment=$deployment] ===")
            for (event in events) when (event) {
                is PromptExecutionEvent.PromptStarted ->
                    add("started: model=${event.modelName}")
                is PromptExecutionEvent.StreamResponded ->
                    add("responded: served=${event.servedModelName}")
                is PromptExecutionEvent.PartReceived -> when (val p = event.part) {
                    is MessagePart.Text -> add("text: ${p.text}")
                    is MessagePart.Thought -> add("thought: ${p.text} [sig=${p.signature?.take(60)}...]")
                    is MessagePart.ToolCall -> add("tool_call: ${p.name}(${p.args})")
                    is MessagePart.InlineData -> add("inline_data: ${p.mimeType}, ${p.sizeBytes} bytes")
                    else -> add("part: $p")
                }
                is PromptExecutionEvent.UsageDelta ->
                    add("usage: input=${event.input} output=${event.output}")
                is PromptExecutionEvent.Completed ->
                    add("completed: finish=${event.finishReason}")
                is PromptExecutionEvent.Failed ->
                    add("FAILED: ${event.error}")
                else -> add("event: $event")
            }
        }
        lines.forEach(::println)
        System.getenv("AZURE_TEST_REPORT_FILE")?.let { path ->
            File(path).appendText(lines.joinToString("\n", postfix = "\n\n"))
        }
    }

    private fun List<PromptExecutionEvent>.textOf(): String = this
        .filterIsInstance<PromptExecutionEvent.PartReceived>()
        .mapNotNull { (it.part as? MessagePart.Text)?.text }
        .joinToString("")

    @Test
    fun `simple prompt returns non-empty text`() = runBlocking {
        val client = AzureOpenAiClient(sdkClient(), DirectGptStrategy(sdkClient()))
        val prompt = buildPrompt {
            llm = testLlm()
            user { text("Say hello in one short sentence.") }
        }

        val events = client.execute(prompt).toList()
        report("simple prompt (direct)", events)
        assertTrue(events.textOf().isNotBlank())
    }

    @Test
    fun `embedding returns a 768-dimensional vector`() = runBlocking {
        val client = AzureOpenAiClient(sdkClient(), DirectGptStrategy(sdkClient()))
        val prompt = Prompt(
            llm = Llm("azure", "embedding", "Text Embedding 3 Large", "3", true, embeddingDeployment),
            settings = EmbeddingPromptSettings(outputDimensionality = 768),
            content = listOf(PromptContent(PromptContentType.USER, listOf(TextPromptPart("dark tense dungeon ambience")))),
        )
        val events = client.execute(prompt).toList()
        report("embedding (direct)", events)
        val embedding = events
            .filterIsInstance<PromptExecutionEvent.PartReceived>()
            .mapNotNull { it.part as? MessagePart.Embedding }
            .single()
        assertEquals(768, embedding.dimension)
        assertTrue(embedding.values.any { it != 0f }, "embedding must not be all-zero")
    }

    /** Minimal hand-written [InstrumentDeclaration] — no `@Instrument` reflection needed. */
    private class NoArgTool : InstrumentDeclaration<String> {
        override val obj: Any get() = this
        override val name = "get_current_time"
        override val description = "Returns the current time as a string"
        override val parameters: List<InstrumentParameter> = emptyList()
        override val returnType: KType = typeOf<String>()
        override suspend fun invoke(args: Map<String, Any?>): String = "12:00"
    }

    @Test
    fun `tool calling round-trips encrypted reasoning into the second request`() = runBlocking {
        val tool = NoArgTool()
        val client = AzureOpenAiClient(sdkClient(), DirectGptStrategy(sdkClient()))
        // An explicit thinking level: without actual reasoning the model returns no reasoning
        // item at all (nothing to replay), so the test must make it think. MEDIUM (not LOW):
        // the server only generates a readable reasoning summary when there is enough actual
        // reasoning to summarise — a trivial LOW turn returns an item with no summary.
        val toolSettings = ToolsPromptSettings(
            tools = listOf(InstrumentsDeclaration.of(tool, listOf(tool))),
            thinkingLevel = ThinkingLevel.MEDIUM,
        )
        val question = "I want to schedule a call exactly 3 hours and 45 minutes from now, " +
            "but only if that lands strictly before 17:30. Find out the current time with the tool, " +
            "then tell me whether the call fits and at what time it would start."
        val prompt = buildPrompt {
            llm = testLlm()
            settings = toolSettings
            user { text(question) }
        }

        // RT1: the model must call the tool; the reasoning signature must ride on the call part.
        val events = client.execute(prompt).toList()
        report("tool calling RT1 (direct)", events)
        val toolCall = events
            .filterIsInstance<PromptExecutionEvent.PartReceived>()
            .mapNotNull { it.part as? MessagePart.ToolCall }
            .first()
        val reasoning = decodeReasoningSignature(toolCall.thoughtSignature)
        println("RT1 reasoning signature: id=${reasoning?.id} encLen=${reasoning?.enc?.length}")
        assertTrue(reasoning?.enc != null, "RT1 tool call must carry a replayable reasoning signature (id+encrypted_content)")

        // Append the exchange to the history the way ToolCallingAgent does (call part carries the
        // signature; result matched positionally).
        val call = org.nobirds.murmuration.prompt.ToolCall(toolCall.name, toolCall.args, toolCall.thoughtSignature, callId = CallIds.mint())
        val historyPrompt = buildPrompt {
            llm = testLlm()
            settings = toolSettings
            user { text(question) }
        }.let {
            Prompt(
                it.llm, it.settings,
                it.content + listOf(
                    PromptContent(PromptContentType.MODEL, listOf(ToolCallPromptPart(call))),
                    PromptContent(PromptContentType.TOOL, listOf(ToolResultPromptPart(call.name, JsonPrimitive("12:00"), callId = CallIds.mint()))),
                ),
            )
        }

        // Direct proof the reasoning item goes onto the wire: the next-request input contains
        // ofReasoning with the original encrypted content immediately before the function call.
        val nextInput = historyPrompt.content.toResponseInput()
        val reasoningIdx = nextInput.indexOfFirst { it.isReasoning() }
        assertTrue(reasoningIdx >= 0, "second request input must contain the replayed reasoning item")
        val replayed = nextInput[reasoningIdx].reasoning().get()
        println("RT2 input: reasoning id=${replayed.id()} encLen=${replayed.encryptedContent().get().length} followedByCall=${nextInput[reasoningIdx + 1].isFunctionCall()}")
        assertTrue(nextInput[reasoningIdx + 1].isFunctionCall(), "reasoning item must immediately precede its function call")

        // RT2 against the live endpoint: the server must ACCEPT the replayed reasoning item
        // (an orphaned/broken pair would 400) and finish the turn.
        val events2 = client.execute(historyPrompt).toList()
        report("tool calling RT2 with reasoning replay (direct)", events2)
        assertTrue(events2.any { it is PromptExecutionEvent.Completed }, "RT2 with the replayed reasoning item must complete")
    }

    @Serializable
    data class Answer(@SchemaInfo("The numeric answer") val value: Int)

    @Test
    fun `structured output produces a typed json response`() = runBlocking {
        val client = AzureOpenAiClient(sdkClient(), DirectGptStrategy(sdkClient()))
        val prompt = buildPrompt {
            llm = testLlm()
            settings = TypedPromptSettings(Answer::class)
            user { text("What is 2 + 2? Reply with the numeric value.") }
        }

        val events = client.execute(prompt).toList()
        report("structured output (direct)", events)
        assertTrue(events.textOf().isNotBlank())
    }

    @Test
    fun `streaming returns non-empty text`() = runBlocking {
        val client = AzureOpenAiClient(sdkClient(), StreamingGptStrategy(sdkClient(), 30.seconds, 120.seconds))
        val prompt = buildPrompt {
            llm = testLlm()
            user { text("Count from 1 to 5.") }
        }

        val events = client.execute(prompt).toList()
        report("streaming", events)
        assertTrue(events.textOf().isNotBlank())
    }

    /** Tiny solid-color square, encoded as PNG bytes — a cheap stand-in for real reference art. */
    private fun solidColorSquare(color: Color, side: Int = 64): ByteArray {
        val image = BufferedImage(side, side, BufferedImage.TYPE_INT_RGB)
        val g = image.createGraphics()
        try {
            g.color = color
            g.fillRect(0, 0, side, side)
        } finally {
            g.dispose()
        }
        return ByteArrayOutputStream().use { out ->
            ImageIO.write(image, "png", out)
            out.toByteArray()
        }
    }

    @Test
    @Disabled("Manual: live multi-reference images.edit on gpt-image-2; run only with explicit approval")
    fun `images edit blends two inline reference images`() = runBlocking {
        val client = AzureOpenAiClient(sdkClient(), DirectGptStrategy(sdkClient()), deploymentResolver = { imageDeployment })
        val redSwatch = solidColorSquare(Color.RED)
        val blueSwatch = solidColorSquare(Color.BLUE)

        val prompt = buildPrompt {
            llm = OpenAiLlms.AZURE_GPT_IMAGE_2
            settings = ImagePromptSettings(size = ImageSize(ImageResolution.SD_512, ImageAspectRatio.SQUARE_1_1))
            user {
                text(
                    "Two reference swatches are attached: a solid red square and a solid blue " +
                        "square. Generate a single small icon showing both colors side by side, " +
                        "split diagonally."
                )
                image(redSwatch, "image/png")
                image(blueSwatch, "image/png")
            }
        }

        // Two inline DataPromptPart images in the request → AzureOpenAiClient.executeImage takes
        // the images.edit branch (images.generate would fire only with zero input images) —
        // this is the multi-reference path TurnImageGenerator's map/token agents rely on.
        val events = client.execute(prompt).toList()
        report("images.edit multi-reference (direct)", events)
        val inlineImage = events
            .filterIsInstance<PromptExecutionEvent.PartReceived>()
            .mapNotNull { it.part as? MessagePart.InlineData }
            .firstOrNull()
        assertTrue(
            inlineImage != null && inlineImage.sizeBytes > 0,
            "images.edit with two reference images must return non-empty inline image bytes",
        )
    }

    @Test
    @Disabled("Manual: live images.generate on gpt-image-2; run only with explicit approval")
    fun `images generate produces an image from a text-only prompt`() = runBlocking {
        val client = AzureOpenAiClient(sdkClient(), DirectGptStrategy(sdkClient()), deploymentResolver = { imageDeployment })

        val prompt = buildPrompt {
            llm = OpenAiLlms.AZURE_GPT_IMAGE_2
            settings = ImagePromptSettings(size = ImageSize(ImageResolution.SD_512, ImageAspectRatio.SQUARE_1_1))
            user {
                text("a simple red circle on white background, flat icon style")
            }
        }

        // Zero inline DataPromptPart images in the request → AzureOpenAiClient.executeImage
        // takes the images.generate branch (images.edit would fire only with at least one input
        // image) — this is the plain text-to-image path.
        val events = client.execute(prompt).toList()
        report("images.generate text-only (direct)", events)
        val inlineImage = events
            .filterIsInstance<PromptExecutionEvent.PartReceived>()
            .mapNotNull { it.part as? MessagePart.InlineData }
            .firstOrNull()
        assertTrue(
            inlineImage != null && inlineImage.sizeBytes > 0,
            "images.generate with a text-only prompt must return non-empty inline image bytes",
        )
        val usage = events.filterIsInstance<PromptExecutionEvent.UsageDelta>().firstOrNull()
        println("images.generate usage: $usage")
    }
}
