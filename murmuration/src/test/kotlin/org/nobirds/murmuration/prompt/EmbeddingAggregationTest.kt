package org.nobirds.murmuration.prompt

import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class EmbeddingAggregationTest {

    @Test
    fun `aggregate produces EmbeddingPromptExecutionResult from an embedding part`() = runBlocking {
        val vec = floatArrayOf(0.1f, -0.2f, 0.3f)
        val executor = PromptExecutor {
            flowOf<PromptExecutionEvent>(
                PromptExecutionEvent.PromptStarted("gemini-embedding-001", "gemini-embedding-001", emptyList()),
                PromptExecutionEvent.UsageDelta(input = InputTokenStatistics(total = 36, cached = 0, cacheWriteShort = 0, cacheWriteLong = 0)),
                PromptExecutionEvent.PartReceived(MessagePart.Embedding(vec.size, vec)),
                PromptExecutionEvent.Completed(),
            )
        }

        val result = executor.execute(emptyPrompt()).aggregate()

        assertTrue(result is EmbeddingPromptExecutionResult, "embedding part must aggregate to embedding result")
        assertEquals(vec.toList(), result.embedding.toList())
        // Statistics ride the common pipe: model name and usage flow into the result.
        assertEquals("gemini-embedding-001", result.statistics.modelName)
        assertEquals(36L, result.statistics.tokens.input.total)
    }

    @Test
    fun `embedding result round-trips through toEventFlow`() = runBlocking {
        val vec = floatArrayOf(1f, 2f, 3f, 4f)
        val original = EmbeddingPromptExecutionResult(
            embedding = vec,
            statistics = PromptExecutionStatistics(modelName = "gemini-embedding-001"),
        )

        val roundTripped = original.toEventFlow().aggregate()

        assertTrue(roundTripped is EmbeddingPromptExecutionResult)
        assertEquals(vec.toList(), roundTripped.embedding.toList())
    }

    @Test
    fun `embed extension extracts the vector`() = runBlocking {
        val vec = floatArrayOf(0.5f, -0.5f, 0.25f)
        val executor = PromptExecutor {
            flowOf<PromptExecutionEvent>(
                PromptExecutionEvent.PromptStarted("gemini-embedding-001", "gemini-embedding-001", emptyList()),
                PromptExecutionEvent.PartReceived(MessagePart.Embedding(vec.size, vec)),
                PromptExecutionEvent.Completed(),
            )
        }

        val out = executor.embed(
            org.nobirds.murmuration.llm.Llm("test", "test", "test", "1", true, "test"),
            "tense tavern brawl",
        )

        assertEquals(vec.toList(), out.toList())
    }

    private fun emptyPrompt(): Prompt = Prompt(
        llm = org.nobirds.murmuration.llm.Llm(
            provider = "test", family = "test", name = "test",
            version = "1", stable = true, model = "test",
        ),
        settings = EmbeddingPromptSettings(),
        content = emptyList(),
    )
}
