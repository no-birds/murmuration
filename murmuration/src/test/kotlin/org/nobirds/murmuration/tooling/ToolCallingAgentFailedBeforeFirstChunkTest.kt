package org.nobirds.murmuration.tooling

import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.agent.AgentExecutionEntry
import org.nobirds.murmuration.agent.PipelineEventListener
import org.nobirds.murmuration.agent.PromptExecutionEntry
import org.nobirds.murmuration.agent.execute
import org.nobirds.murmuration.agent.executePipeline
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PromptCachingProvider
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Failed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.PromptMessage
import org.nobirds.murmuration.prompt.StreamDiagnostics
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * the REAL failure shape of a Gemini round-trip that dies before its first chunk —
 * the executor emits `PromptStarted` (before sending), then `Failed`, then THROWS the same exception
 * (`GeminiPromptExecutor`: `emit(Failed(mapped)); throw mapped`). Every other core test emits `Failed`
 * WITHOUT the following `throw`, which hides that `executeTurn` assembles `stats` only after `collect`
 * returns normally — the `throw` bypasses it and `partialStats` is lost.
 *
 * Both tests are RED on HEAD, by design (they pin the defect, they are not exploration):
 * 1. [partialStats survive a Failed followed by a throw] — `executeTurn` has `try/finally` but no
 *    `catch`, so the thrown exception leaves `partialStats = null` and the FAILURE row is written
 *    with `cacheId = null`, `requestMessages = []`.
 * 2. [onStreamStarted does not fire for a round-trip the provider never answered] — `onStreamStarted`
 *    is dispatched on `PromptStarted` (request sent), not on the provider's first response.
 *    Green in S2 (dispatch on `StreamResponded`) together with 1 (stats built in a `catch`).
 *
 * The fake emits `PromptStarted` already in its post-S2 position (before any chunk), which is the
 * only place the design allows it.
 */
class ToolCallingAgentFailedBeforeFirstChunkTest {

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    class NoopTool {
        @Instrument("Always succeeds.")
        fun ping(): String = "pong"
    }

    private class CapturingListener : PipelineEventListener {
        val entries = mutableListOf<PromptExecutionEntry>()
        val streamStarted = mutableListOf<Pair<UUID, String>>()

        override suspend fun onEntryReported(pipelineId: UUID, entry: AgentExecutionEntry) {
            if (entry is PromptExecutionEntry) entries += entry
        }

        override suspend fun onStreamStarted(pipelineId: UUID, executionId: UUID, modelName: String) {
            streamStarted += executionId to modelName
        }
    }

    private val requestMessages = listOf(PromptMessage(PromptContentType.USER, listOf(MessagePart.Text("the request as sent"))))

    /** `PromptStarted` (cache + request known) -> `Failed` (0 chunks, TTFB timeout) -> `throw` the same error. */
    private fun silentThenThrowExecutor(): PromptExecutor = PromptExecutor {
        flow {
            emit(
                PromptStarted(
                    modelName = "test-model",
                    requestedModelName = "test-model",
                    requestMessages = requestMessages,
                    cacheId = "cachedContents/test-cache",
                    cacheStats = PromptCachingProvider.CacheStatistics(tokenCount = 4242),
                ),
            )
            val diag = StreamDiagnostics(
                chunkCount = 0, totalElapsedMs = 60_000, sinceLastChunkMs = 60_000,
                partsText = 0, partsThought = 0, partsToolCall = 0, firstChunkMs = null,
            )
            val error = TransientException(TransientReason.NETWORK_TIMEOUT, "no first event within 60s", null)
            error.streamDiagnostics = diag
            emit(Failed(error, streamDiagnostics = diag))
            throw error
        }
    }

    private fun runFailing(listener: CapturingListener): TransientException = runBlocking {
        assertFailsWith<TransientException> {
            executePipeline(pipelineEventListener = listener) {
                execute(
                    toolCallingAgent<Unit, NoopTool>(
                        llm = fakeLlm, executor = silentThenThrowExecutor(), name = "SilentAgent", maxTurns = 1,
                    ) { _, _, _ -> user { text("test prompt") } },
                    ToolCallingInput(Unit, NoopTool()),
                )
            }
        }
    }

    @Test
    fun `partialStats survive a Failed followed by a throw - cacheId and request are not lost`() {
        val listener = CapturingListener()

        val thrown = runFailing(listener)

        val partial = assertNotNull(thrown.partialStats, "a round-trip that reached the wire must carry partialStats out")
        assertEquals("cachedContents/test-cache", partial.cacheId)
        assertEquals(4242L, partial.cachedTokenCount)
        assertEquals(requestMessages, partial.requestMessages)
        assertEquals(0L, partial.tokens.totalTokens, "provider counted nothing: tokens stay 0 (true, not an artefact)")
        assertEquals(0, assertNotNull(partial.streamDiagnostics).chunkCount)

        // The FAILURE record rebuilt from the exception carries the same — and there is exactly one.
        val failure = listener.entries.single()
        assertEquals(false, failure.statistics.success)
        assertEquals("cachedContents/test-cache", failure.statistics.cacheId)
        assertTrue(failure.statistics.requestMessages.isNotEmpty())
        assertNull(failure.statistics.streamDiagnostics?.firstChunkMs)
    }

    @Test
    fun `onStreamStarted does not fire for a round-trip the provider never answered`() {
        val listener = CapturingListener()

        runFailing(listener)

        assertEquals(
            0, listener.streamStarted.size,
            "PromptStarted means 'request sent', not 'provider responded'; saw ${listener.streamStarted}",
        )
    }
}
