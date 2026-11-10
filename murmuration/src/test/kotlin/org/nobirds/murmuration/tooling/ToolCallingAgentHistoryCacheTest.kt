package org.nobirds.murmuration.tooling

import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.agent.execute
import org.nobirds.murmuration.agent.executePipeline
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.llm.LlmProviders
import org.nobirds.murmuration.prompt.CallIds
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptCacheScope
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** The agent marks its own history only by the model's cache kind: tail mark for `anthropic`, nothing otherwise. */
class ToolCallingAgentHistoryCacheTest {

    class PingTool {
        @Instrument("Ping the test tool.")
        fun ping(): String = "pong"
    }

    private fun llm(provider: String) = Llm(
        provider = provider, family = "f", name = "n", version = "1", stable = true, model = "m",
    )

    /** Cache scopes of the agent's own history (MODEL and TOOL contents) in the second round-trip's prompt. */
    private fun historyScopes(provider: String): Set<PromptCacheScope> = runBlocking {
        var calls = 0
        var captured: Prompt? = null
        val executor = PromptExecutor { prompt ->
            flow {
                calls++
                emit(PromptStarted(modelName = "m", requestedModelName = "m", requestMessages = emptyList()))
                if (calls == 1) {
                    emit(PartReceived(MessagePart.Thought("t", "s")))
                    emit(PartReceived(MessagePart.Text("x")))
                    emit(PartReceived(MessagePart.ToolCall("ping", emptyMap(), null, callId = CallIds.mint())))
                } else {
                    captured = prompt
                    emit(PartReceived(MessagePart.Text("done")))
                }
                emit(Completed(finishReason = "completed"))
            }
        }
        executePipeline {
            val agent = toolCallingAgent<Unit, PingTool>(
                llm = llm(provider), executor = executor, name = "HistoryCacheAgent", maxTurns = 3,
            ) { _, _, _ -> user { text("test prompt") } }
            execute(agent, ToolCallingInput(Unit, PingTool()))
        }
        val history = checkNotNull(captured).content
            .filter { it.type == PromptContentType.MODEL || it.type == PromptContentType.TOOL }
        assertTrue(history.sumOf { it.parts.size } >= 4)
        history.flatMap { it.parts }.map { it.cacheStrategy.scope }.toSet()
    }

    @Test
    fun `anthropic history carries the tail mark on every agent part`() {
        assertEquals(setOf(PromptCacheScope.TTL_BASED), historyScopes(LlmProviders.ANTHROPIC))
    }

    @Test
    fun `gemini and azure history stays NO_CACHE`() {
        assertEquals(setOf(PromptCacheScope.NO_CACHE), historyScopes("google"))
        assertEquals(setOf(PromptCacheScope.NO_CACHE), historyScopes("azure"))
    }
}
