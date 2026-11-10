package org.nobirds.murmuration.gemini

import com.google.genai.Client
import com.google.genai.types.Content
import com.google.genai.types.GenerateContentConfig
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.ImageAspectRatio
import org.nobirds.murmuration.prompt.ImageResolution
import org.nobirds.murmuration.prompt.ImageSize
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptCachingProvider
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptMessage
import org.nobirds.murmuration.prompt.ThinkingLevel
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.image
import org.nobirds.murmuration.prompt.system
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.prompt.withImageSize
import org.nobirds.murmuration.prompt.withThinkingLevel
import kotlin.jvm.optionals.getOrNull
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * Checks that `thinkingLevel`, attached by the pipeline decorator [withThinkingLevel] to an
 * image prompt, actually reaches `GenerateContentConfig.thinkingConfig` at the genai
 * SDK boundary — WITHOUT a live call. We capture the config via a stub [GeminiResponseStrategy].
 *
 * Covers both production image models (the senior `gemini-3.1-flash-image` — player portraits,
 * book covers; and the cheap `gemini-3.1-flash-lite-image` — NPC art/maps) on both providers,
 * and both paths: pure generation (text-only input) and image-EDIT (an input image +
 * text, like deriving a token from a portrait). Regression barrier:
 *  - for lite, HIGH is declared in the catalog ([GeminiLlms]) and passes `clampThinkingLevel`;
 *  - the senior has NO thinking declaration → `clampThinkingLevel` = pass-through (absence of
 *    a declaration means "no known restriction", NOT "no thinking"), so HIGH also
 *    arrives without silent degradation. If someone attaches a restrictive
 *    `level(...)` without HIGH to the senior, or the decorator starts clamping — the test catches the HIGH degradation.
 */
class GeminiImageThinkingLevelTest {

    /** Captures the [GenerateContentConfig] built by the executor instead of a real round-trip. */
    private class CapturingStrategy : GeminiResponseStrategy {
        var captured: GenerateContentConfig? = null

        override fun execute(
            prompt: Prompt,
            modelName: String,
            contents: List<Content>,
            config: GenerateContentConfig,
            cacheResult: PromptCachingProvider.CacheResult?,
            requestMessages: List<PromptMessage>,
            requestedModelName: String,
        ): Flow<PromptExecutionEvent> = flow {
            captured = config
            emit(PromptExecutionEvent.Completed())
        }
    }

    private val size = ImageSize(ImageResolution.HD_1024, ImageAspectRatio.SQUARE_1_1)

    // Gemini API client without a network: apiKey mode only remembers the key, HTTP goes out at the moment of
    // the call, which does not exist here (the round-trip is intercepted by CapturingStrategy).
    private fun dummyClient(): Client = Client.builder().apiKey("test-key-not-used").build()

    private fun capturedThinkingLevelFor(prompt: Prompt): com.google.genai.types.ThinkingLevel? {
        val strategy = CapturingStrategy()
        val client = GeminiClient(dummyClient(), GeminiWireKeyLayout.NONE, responseStrategy = strategy)
        // Production order: the agent attaches withImageSize when assembling the prompt, and the pipeline
        // wraps the executor with the withThinkingLevel(HIGH) decorator.
        val executor = client.withThinkingLevel(ThinkingLevel.HIGH)
        runBlocking { executor.execute(prompt.withImageSize(size)).collect {} }
        val config = assertNotNull(strategy.captured, "strategy did not capture a config")
        return config.thinkingConfig().getOrNull()?.thinkingLevel()?.getOrNull()
    }

    private fun generatePrompt(model: Llm): Prompt = buildPrompt {
        llm = model
        system { text("You are an image generator.") }
        user { text("A simple forest clearing with a campfire.") }
    }

    private fun editPrompt(model: Llm): Prompt = buildPrompt {
        llm = model
        system { text("You are an image editor.") }
        // The input image = the signature of the image-EDIT path (deriving a token from a portrait).
        user {
            image(ByteArray(16) { it.toByte() }, "image/png")
            text("Turn this portrait into a round token, keep the same face.")
        }
    }

    private val expected = ThinkingLevel.HIGH.toGemini()

    private val models: List<Pair<String, Llm>> = listOf(
        "vertex senior gemini-3.1-flash-image" to GeminiLlms.Vertex.VERTEX_GEMINI_3_1_FLASH_IMAGE,
        "vertex lite gemini-3.1-flash-lite-image" to GeminiLlms.Vertex.VERTEX_GEMINI_3_1_FLASH_LITE_IMAGE,
        "google senior gemini-3.1-flash-image" to GeminiLlms.Gemini.GEMINI_3_1_FLASH_IMAGE,
        "google lite gemini-3.1-flash-lite-image" to GeminiLlms.Gemini.GEMINI_3_1_FLASH_LITE_IMAGE,
    )

    @Test
    fun `thinking HIGH reaches config on the generate path for both image models`() {
        for ((label, model) in models) {
            val actual = capturedThinkingLevelFor(generatePrompt(model))
            assertEquals(expected, actual, "generate path, $label: thinkingLevel HIGH did not reach the request config")
        }
    }

    @Test
    fun `thinking HIGH reaches config on the image-edit path for both image models`() {
        for ((label, model) in models) {
            val actual = capturedThinkingLevelFor(editPrompt(model))
            assertEquals(expected, actual, "edit path, $label: thinkingLevel HIGH did not reach the request config")
        }
    }
}
