package org.nobirds.murmuration.tooling

import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.ExecutionTextPart
import org.nobirds.murmuration.prompt.NoOpImagePromptExecutor
import org.nobirds.murmuration.prompt.PromptExecutionStatistics
import org.nobirds.murmuration.prompt.TextPromptExecutionResult
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.toEventFlow
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.llm.Llm
import kotlin.test.assertTrue

/**
 * core side: event-order contract of the core emitters of `PromptStarted`
 * ([toEventFlow] — replay of a ready result for single-shot agents, and [NoOpImagePromptExecutor]).
 * One `PromptStarted`, first; `StreamResponded` once right after it, before the parts.
 */
class ProviderEventOrderCoreTest {

    private val result = TextPromptExecutionResult(
        statistics = PromptExecutionStatistics(modelName = "m", requestedModelName = "m"),
        textParts = listOf(ExecutionTextPart("hello", isThought = false)),
    )

    private val imagePrompt = buildPrompt {
        llm = Llm(provider = "test", family = "test", name = "test", version = "1", stable = true, model = "noop")
        user { text("draw") }
    }

    @Test
    fun `toEventFlow - one PromptStarted first`() = runBlocking {
        val events = result.toEventFlow().toList()
        assertPromptStartedFirstAndOnce(events)
        assertTrue(events.size > 1, "events=${events.names()}")
    }

    @Test
    fun `NoOpImagePromptExecutor - one PromptStarted first`() = runBlocking {
        assertPromptStartedFirstAndOnce(NoOpImagePromptExecutor.execute(imagePrompt).toList())
    }

    @Test
    fun `toEventFlow - StreamResponded once right after PromptStarted`() = runBlocking {
        assertStreamRespondedOnce(result.toEventFlow().toList())
    }

    @Test
    fun `NoOpImagePromptExecutor - StreamResponded once right after PromptStarted`() = runBlocking {
        assertStreamRespondedOnce(NoOpImagePromptExecutor.execute(imagePrompt).toList())
    }
}
