package org.nobirds.murmuration.gemini

import com.google.genai.Client
import com.google.genai.types.HttpOptions
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.mapNotNull
import kotlinx.coroutines.runBlocking
import java.util.UUID
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.ImageAspectRatio
import org.nobirds.murmuration.prompt.ImageResolution
import org.nobirds.murmuration.prompt.ImageSize
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.image
import org.nobirds.murmuration.prompt.system
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.prompt.withImageSize
import org.nobirds.murmuration.llm.Llm
import java.io.File

class GeminiImageGenerationTest {

    @Test
    @Disabled("Direct image generation test for manual run")
    fun testMapGeneration() {
        val vertexProjectId = System.getenv("VERTEX_AI_PROJECT_ID") ?: "my-project-id"

        val sdkClient =
            Client.builder()
                .vertexAI(true)
                .project(vertexProjectId)
                .location("global")
                .httpOptions(HttpOptions.builder().apiVersion("v1").build())
                .build()

        val client = GeminiClient(sdkClient, GeminiWireKeyLayout.NONE)

        // Try different sizes that might occur in production
        val sizesToTest = listOf(
            ImageSize(ImageResolution.SD_512, ImageAspectRatio.SQUARE_1_1),
            ImageSize(ImageResolution.HD_1024, ImageAspectRatio.SQUARE_1_1),
            ImageSize(ImageResolution.XHD_2048, ImageAspectRatio.SQUARE_1_1),
            ImageSize(ImageResolution.UHD_4096, ImageAspectRatio.SQUARE_1_1),
        )

        val results = mutableListOf<String>()
        for (size in sizesToTest) {
            val prompt = buildPrompt {
                llm = Llm("vertex", "imagen", "Imagen 3.1 Flash", "3.1", false, "gemini-3.1-flash-image")
                system {
                    text("You are a map generator. Generate a top-down tactical map.")
                }
                user {
                    text("A simple forest clearing with a small campfire in the middle. 24x24 grid.\"")
                }
            }

            runBlocking {
                try {
                    val blob = client.execute(prompt.withImageSize(size))
                        .filterIsInstance<PromptExecutionEvent.PartReceived>()
                        .mapNotNull { it.part as? MessagePart.InlineData }
                        .first()
                    val bytes = blob.data ?: error("InlineData.data is null")
                    results.add("Size ${size}: OK (${bytes.size} bytes)")
                } catch (e: Exception) {
                    results.add("Size ${size}: FAIL - ${e.message}")
                }
            }
        }

        results.forEach { println(it) }
        if (results.any { it.contains("FAIL") }) {
            error("Some sizes failed:\n${results.joinToString("\n")}")
        }
    }

    @Test
    @Disabled("Direct image generation test for manual run")
    fun testDirectImageGeneration() = runBlocking {
        val vertexProjectId = System.getenv("VERTEX_AI_PROJECT_ID") ?: "my-project-id"

        println("Generation for project: $vertexProjectId")

        val sdkClient = Client.builder()
            .vertexAI(true)
            .project(vertexProjectId)
            .location("global")
            .httpOptions(HttpOptions.builder().apiVersion("v1").build())
            .build()

        val client = GeminiClient(sdkClient, GeminiWireKeyLayout.NONE)

        val prompt = buildPrompt {
            llm = Llm("vertex", "imagen", "Imagen 3.1 Flash", "3.1", false, "gemini-3.1-flash-image")

            user {
                image(File("3f7dccb4-f027-4393-5068-c269062f5741.png").readBytes(), "image/png")
                text("Make a token from the image, show its face and shoulders, make a round frame around the whole image")
            }
        }

        try {
            println("Generation started")
            val blob = client.execute(
                prompt.withImageSize(ImageSize(ImageResolution.SD_512, ImageAspectRatio.SQUARE_1_1))
            )
                .filterIsInstance<PromptExecutionEvent.PartReceived>()
                .mapNotNull { it.part as? MessagePart.InlineData }
                .first()
            val bytes = blob.data ?: error("InlineData.data is null")
            File(UUID.randomUUID().toString() + ".png").writeBytes(bytes)
            println("Direct generation OK: ${bytes.size} bytes")
        } catch (e: Exception) {
            println("Direct generation FAILED: ${e.message}")
            throw e
        }
    }

}
