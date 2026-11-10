package org.nobirds.murmuration.tooling

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.agent.AgentExecutionEntry
import org.nobirds.murmuration.agent.PipelineEventListener
import org.nobirds.murmuration.agent.PromptExecutionEntry
import org.nobirds.murmuration.agent.execute
import org.nobirds.murmuration.agent.executePipeline
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Guard: cooperative cancellation inside a round-trip goes past the
 * `PromptExecutionException` handling of `executeTurn` — the cancellation propagates as is and no
 * failure statistics are assembled for it (a cancelled round-trip is not a provider failure).
 * Green on HEAD; must stay green when the statistics assembly moves (S7).
 */
class ToolCallingAgentCancellationTest {

    private val fakeLlm = Llm(provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model")

    class NoopTool {
        @Instrument("Always succeeds.")
        fun ping(): String = "pong"
    }

    private class EntryListener : PipelineEventListener {
        val entries = mutableListOf<PromptExecutionEntry>()
        override suspend fun onEntryReported(pipelineId: UUID, entry: AgentExecutionEntry) {
            if (entry is PromptExecutionEntry) entries += entry
        }
    }

    @Test
    fun `cancellation inside a round-trip propagates and assembles no failure statistics`() {
        val listener = EntryListener()
        val executor = PromptExecutor {
            flow {
                emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
                throw CancellationException("cancelled by the caller")
            }
        }

        val thrown = runBlocking {
            assertFailsWith<CancellationException> {
                executePipeline(pipelineEventListener = listener) {
                    execute(
                        toolCallingAgent<Unit, NoopTool>(llm = fakeLlm, executor = executor, name = "CancelAgent", maxTurns = 1) { _, _, _ ->
                            user { text("test prompt") }
                        },
                        ToolCallingInput(Unit, NoopTool()),
                    )
                }
            }
        }

        assertEquals("cancelled by the caller", thrown.message)
        assertTrue(
            listener.entries.none { it.statistics.errorCode != null || it.statistics.cacheId != null || it.statistics.requestMessages.isNotEmpty() },
            "a cancelled round-trip must not carry assembled failure statistics; entries=${listener.entries.map { it.statistics }}",
        )
    }
}
