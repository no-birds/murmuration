package org.nobirds.murmuration.tooling

import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.agent.PipelineEventListener
import org.nobirds.murmuration.agent.execute
import org.nobirds.murmuration.agent.executePipeline
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Failed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutionEvent.StreamResponded
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * `onStreamStarted` must fire the moment the
 * stream's first event (`PromptStarted`) arrives — NOT after `TurnResult` is assembled. On an
 * obryv-stream round-trip `executeTurn` throws its `collectedFailure` before assembly is ever
 * reached (`Failed` after `PromptStarted`, no `Completed`), so a dispatch placed after assembly
 * would silently miss exactly the round-trips this event exists to make visible (K4: "80 starts
 * against 78 records"). This test is written so a regression to "dispatch after TurnResult" fails
 * it: on this fixture that redaction never calls `onStreamStarted` at all.
 */
class ToolCallingAgentStreamStartedTest {

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    class NoopTool {
        @Instrument("Always succeeds.")
        fun ping(): String = "pong"
    }

    private class CapturingListener : PipelineEventListener {
        val streamStarted = mutableListOf<Triple<UUID, UUID, String>>()

        override suspend fun onStreamStarted(pipelineId: UUID, executionId: UUID, modelName: String) {
            streamStarted += Triple(pipelineId, executionId, modelName)
        }
    }

    /** `PromptStarted` -> one text part -> `Failed`, mid-stream — no `Completed` ever arrives. */
    private fun obryvStreamExecutor(): PromptExecutor = PromptExecutor {
        flow {
            emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
            emit(StreamResponded(servedModelName = "test-model"))
            emit(PartReceived(MessagePart.Text("partial thought before the drop")))
            emit(Failed(TransientException(TransientReason.INCOMPLETE_RESPONSE, "stream dropped", null)))
        }
    }

    @Test
    fun `onStreamStarted fires exactly once on a round-trip that breaks before TurnResult is ever assembled`() = runBlocking {
        val tool = NoopTool()
        val listener = CapturingListener()

        assertFailsWith<TransientException> {
            executePipeline(pipelineEventListener = listener) {
                execute(
                    toolCallingAgent<Unit, NoopTool>(
                        llm = fakeLlm, executor = obryvStreamExecutor(), name = "StreamStartAgent", maxTurns = 1,
                    ) { _, _, _ -> user { text("test prompt") } },
                    ToolCallingInput(Unit, tool),
                )
            }
        }

        assertEquals(
            1, listener.streamStarted.size,
            "the stream produced exactly one PromptStarted before it broke — onStreamStarted must fire " +
                "exactly once even though TurnResult is never assembled on this path; saw ${listener.streamStarted}",
        )
        assertEquals("test-model", listener.streamStarted.single().third)
    }

    /** Happy path: still fires exactly once, alongside the successful completion. */
    private fun happyPathExecutor(): PromptExecutor = PromptExecutor {
        flow {
            emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
            emit(StreamResponded(servedModelName = "test-model"))
            emit(PartReceived(MessagePart.Text("done")))
            emit(org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed(finishReason = "STOP"))
        }
    }

    @Test
    fun `onStreamStarted fires exactly once on a normal completed round-trip`() = runBlocking {
        val tool = NoopTool()
        val listener = CapturingListener()

        executePipeline(pipelineEventListener = listener) {
            execute(
                toolCallingAgent<Unit, NoopTool>(
                    llm = fakeLlm, executor = happyPathExecutor(), name = "HappyAgent", maxTurns = 1,
                ) { _, _, _ -> user { text("test prompt") } },
                ToolCallingInput(Unit, tool),
            )
        }

        assertEquals(1, listener.streamStarted.size)
    }
}
