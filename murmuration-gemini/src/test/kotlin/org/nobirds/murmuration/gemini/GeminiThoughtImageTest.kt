package org.nobirds.murmuration.gemini

import com.google.genai.types.Blob
import com.google.genai.types.Candidate
import com.google.genai.types.Content
import com.google.genai.types.GenerateContentResponse
import com.google.genai.types.Part
import kotlinx.coroutines.flow.asFlow
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.BinaryDataStorage
import org.nobirds.murmuration.prompt.ImagePromptExecutionResult
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.aggregate
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.prompt.withBinaryDataStorage
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNull

/**
 * An image generated with thinking arrives as a draft `inlineData` flagged `thought=true` BEFORE the final
 * image. The draft must neither be emitted nor stored, or it wins the aggregator's "first URI wins" rule.
 *
 * Canonical sample: parts of a Gemini response -> `toEventOrNull` (the single conversion point shared by the
 * direct and the streaming paths) -> `withBinaryDataStorage` -> `aggregate`.
 */
class GeminiThoughtImageTest {

    private val draft = "draft".toByteArray()
    private val final = "final".toByteArray()

    private fun image(bytes: ByteArray, thought: Boolean?): Part {
        val builder = Part.builder().inlineData(Blob.builder().mimeType("image/png").data(bytes).build())
        thought?.let { builder.thought(it) }
        return builder.build()
    }

    private fun text(value: String, thought: Boolean) = Part.builder().text(value).thought(thought).build()

    /** Runs [parts] through the mapping, a recording storage and the aggregator; returns result and saved payloads. */
    private fun run(parts: List<Part>): Pair<ImagePromptExecutionResult.File, List<String>> = runBlocking {
        val response = GenerateContentResponse.builder()
            .candidates(listOf(Candidate.builder().content(Content.builder().parts(parts).build()).build()))
            .build()
        val events = response.allParts().mapNotNull { it.toEventOrNull() }
        val saved = mutableListOf<String>()
        val storage = BinaryDataStorage { bytes, mimeType, name ->
            saved += String(bytes)
            MessagePart.File(uri = "mem://${saved.size}", mimeType = mimeType, name = name)
        }
        val prompt = buildPrompt {
            llm = Llm("vertex", "imagen", "Imagen", "1", false, "gemini-3.1-flash-image")
            user { text("draw") }
        }
        val result = PromptExecutor { events.asFlow() }.withBinaryDataStorage(storage).execute(prompt).aggregate()
        assertIs<ImagePromptExecutionResult.File>(result) to saved
    }

    @Test
    fun `thought draft image is dropped and the final image is the result`() {
        val (result, saved) = run(listOf(image(draft, thought = true), text("describing", thought = false), image(final, thought = null)))

        assertEquals(listOf("final"), saved)
        assertEquals("mem://1", result.uri)
    }

    @Test
    fun `thought draft image maps to no event`() {
        assertNull(image(draft, thought = true).toEventOrNull())
        assertIs<PartReceived>(image(draft, thought = false).toEventOrNull())
    }

    @Test
    fun `single image without thought works as before`() {
        val (result, saved) = run(listOf(image(final, thought = null)))

        assertEquals(listOf("final"), saved)
        assertEquals("mem://1", result.uri)
        assertEquals("image/png", result.mimeType)
    }
}
