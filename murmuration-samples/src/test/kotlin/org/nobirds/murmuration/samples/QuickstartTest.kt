package org.nobirds.murmuration.samples

import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.agent.execute
import org.nobirds.murmuration.agent.executePipeline
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.CallIds
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutor
import kotlin.test.assertEquals

/** Runs the quickstart agent against a scripted executor: no API key, no network. */
class QuickstartTest {

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    /** The scripted "model": asks for `add(2, 3)` once, then stops calling tools. */
    private fun scriptedExecutor(): PromptExecutor {
        var round = 0
        return PromptExecutor {
            val current = round++
            flow {
                emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
                if (current == 0) {
                    val args = mapOf<String, JsonElement>("a" to JsonPrimitive(2), "b" to JsonPrimitive(3))
                    emit(PartReceived(MessagePart.ToolCall("add", args, callId = CallIds.mint())))
                }
                emit(Completed(finishReason = "STOP"))
            }
        }
    }

    @Test
    fun `the quickstart agent calls the tool and returns its result`() = runBlocking {
        val agent = calculatorAgent(fakeLlm, scriptedExecutor())

        val sum = executePipeline { execute(agent, "What is 2 + 3?") }

        assertEquals(5, sum)
    }
}
