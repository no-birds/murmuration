package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.prompt.CallIds
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.agent.execute
import org.nobirds.murmuration.agent.executePipeline
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.ToolResultPromptPart
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * A single model response can carry several independent tool calls in one batch. One call
 * failing must NOT abort the round-trip or block its siblings: [ToolsExecutor.executeCall]
 * catches per-call exceptions into an "Error: ..." result string, and
 * [SequentialToolCallScheduler] processes each queued call independently.
 *
 * [ToolCallSchedulerTest] already covers the scheduler's isolation in isolation (calls are
 * submitted manually, one after another, by the test itself). This test instead drives the
 * real [ToolCallingAgent] with a fake [PromptExecutor] that emits BOTH tool calls from the
 * SAME model response — the actual shape of a batch — to pin the guarantee at the layer where
 * a model would experience it.
 */
class ToolCallingAgentBatchIsolationTest {

    class TwoCallTool {
        var okCallCount = 0
            private set

        @Instrument("Always throws — simulates a broken call in the batch.")
        fun fails(@InstrumentParam("Any value") value: String): String {
            throw IllegalStateException("boom: $value")
        }

        @Instrument("Always succeeds — simulates a normal, independent call in the same batch.")
        fun ok(@InstrumentParam("Any value") value: String): String {
            okCallCount++
            return "OK: $value"
        }
    }

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    /**
     * First round-trip emits TWO tool calls in one response (`fails` then `ok`), mirroring a
     * real model batch. Second round-trip emits plain text so the (non-`Stateful`) agent winds
     * down cleanly at `maxTurns = 2` without repeating the batch.
     */
    private fun batchExecutor(captured: MutableList<Prompt>): PromptExecutor = PromptExecutor { prompt ->
        flow {
            val turn = captured.size
            captured += prompt
            emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
            if (turn == 0) {
                emit(PartReceived(MessagePart.ToolCall("fails", mapOf<String, JsonElement>("value" to JsonPrimitive("x")), callId = CallIds.mint())))
                emit(PartReceived(MessagePart.ToolCall("ok", mapOf<String, JsonElement>("value" to JsonPrimitive("y")), callId = CallIds.mint())))
            } else {
                emit(PartReceived(MessagePart.Text("done")))
            }
            emit(Completed(finishReason = "STOP"))
        }
    }

    private fun Prompt.toolResultTexts(): List<String> =
        content.flatMap { it.parts }
            .filterIsInstance<ToolResultPromptPart>()
            .map { (it.result as? JsonPrimitive)?.content ?: it.result.toString() }

    @Test
    fun `one failing call in a batch does not block its sibling call from executing`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = TwoCallTool()
        val agent = toolCallingAgent<Unit, TwoCallTool>(
            llm = fakeLlm, executor = batchExecutor(captured), name = "BatchAgent", maxTurns = 2,
        ) { _, _, _ -> user { text("test prompt") } }

        executePipeline { execute(agent, ToolCallingInput(Unit, tool)) }

        assertEquals(1, tool.okCallCount, "sibling call must have executed despite `fails` throwing in the same batch")

        // The second round-trip's prompt carries the first round-trip's tool results.
        val rt1Results = captured[1].toolResultTexts()
        assertTrue(
            rt1Results.any { it.startsWith("Error:") && it.contains("boom") },
            "failing call's error must be isolated to its own result; saw: $rt1Results",
        )
        assertTrue(
            rt1Results.any { it == "OK: y" },
            "sibling call's own result must be intact, unaffected by its neighbor failing; saw: $rt1Results",
        )
    }
}
