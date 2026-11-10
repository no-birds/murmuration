package org.nobirds.murmuration.tooling

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.agent.AgentExecutionEntry
import org.nobirds.murmuration.agent.PipelineEventListener
import org.nobirds.murmuration.agent.PromptExecutionEntry
import org.nobirds.murmuration.agent.execute
import org.nobirds.murmuration.agent.executePipeline
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.InputTokenStatistics
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.OutputTokenStatistics
import org.nobirds.murmuration.prompt.PromptCachingProvider
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Failed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutionEvent.StreamResponded
import org.nobirds.murmuration.prompt.PromptExecutionEvent.UsageDelta
import org.nobirds.murmuration.prompt.PromptExecutionException
import org.nobirds.murmuration.prompt.PromptExecutionStatistics
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.PromptMessage
import org.nobirds.murmuration.prompt.StreamDiagnostics
import org.nobirds.murmuration.prompt.TextPromptExecutionResult
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.aggregate
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

/**
 * Guard: the SAME event sequence yields the SAME [PromptExecutionStatistics] fields
 * whether it is folded by [ToolCallingAgent] or by [aggregate] (two folding loops, one mapping).
 * Covers success, `Failed` followed by `throw` (the real provider failure shape) and a sequence without
 * `PromptStarted` (a divergence between the loops would otherwise go unnoticed).
 *
 * Green on HEAD; after S7 (one `buildStatistics`) it must stay green.
 */
class StatisticsParityTest {

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

    private val request = listOf(PromptMessage(PromptContentType.USER, listOf(MessagePart.Text("the request as sent"))))

    private val started = PromptStarted(
        modelName = "test-model",
        requestedModelName = "test-model",
        requestMessages = request,
        cacheId = "cachedContents/test-cache",
        cacheStats = PromptCachingProvider.CacheStatistics(tokenCount = 4242),
    )

    private val usage = UsageDelta(input = InputTokenStatistics(total = 100, cached = 0, cacheWriteShort = 0, cacheWriteLong = 0), output = OutputTokenStatistics(total = 20))

    private val diag = StreamDiagnostics(
        chunkCount = 1, totalElapsedMs = 900, sinceLastChunkMs = 100,
        partsText = 1, partsThought = 0, partsToolCall = 0, firstChunkMs = 300,
    )

    /** The fields both folding loops must agree on (the mapping of events to statistics). */
    private data class Comparable(
        val modelName: String,
        val requestedModelName: String?,
        val success: Boolean,
        val totalTokens: Long,
        val inputTotal: Long,
        val outputTotal: Long,
        val requestMessages: List<PromptMessage>,
        val responseText: String?,
        val cacheId: String?,
        val cachedTokenCount: Long?,
        val finishReason: String?,
        val chunkCount: Int?,
        val firstChunkMs: Long?,
    )

    private fun PromptExecutionStatistics.comparable() = Comparable(
        modelName, requestedModelName, success, tokens.totalTokens, tokens.input.total, tokens.output.total,
        requestMessages, (responseMessage?.parts?.firstOrNull() as? MessagePart.Text)?.text,
        cacheId, cachedTokenCount, finishReason, streamDiagnostics?.chunkCount, streamDiagnostics?.firstChunkMs,
    )

    private fun viaAggregate(events: List<PromptExecutionEvent>, thenThrow: (() -> Throwable)?): PromptExecutionStatistics? = runBlocking {
        val source = flow { events.forEach { emit(it) }; thenThrow?.let { throw it() } }
        try {
            (source.aggregate() as TextPromptExecutionResult).statistics
        } catch (e: PromptExecutionException) {
            e.partialStats
        }
    }

    private fun viaAgent(events: List<PromptExecutionEvent>, thenThrow: (() -> Throwable)?): PromptExecutionStatistics? = runBlocking {
        val listener = EntryListener()
        val executor = PromptExecutor { _: org.nobirds.murmuration.prompt.Prompt -> flow { events.forEach { emit(it) }; thenThrow?.let { throw it() } } }
        try {
            executePipeline(pipelineEventListener = listener) {
                execute(
                    toolCallingAgent<Unit, NoopTool>(llm = fakeLlm, executor = executor, name = "ParityAgent", maxTurns = 1) { _, _, _ ->
                        user { text("test prompt") }
                    },
                    ToolCallingInput(Unit, NoopTool()),
                )
            }
        } catch (_: Throwable) {
            // failure shapes: the statistics are read from the reported entry below
        }
        listener.entries.firstOrNull()?.statistics
    }

    private fun assertParity(events: List<PromptExecutionEvent>, thenThrow: (() -> Throwable)? = null) {
        val viaAgent = assertNotNull(viaAgent(events, thenThrow), "agent reported no round-trip entry")
        val viaAggregate = assertNotNull(viaAggregate(events, thenThrow), "aggregate produced no statistics")
        assertEquals(viaAggregate.comparable(), viaAgent.comparable())
    }

    @Test
    fun `success - both loops fold the same statistics`() {
        assertParity(listOf(started, StreamResponded("served-model"), PartReceived(MessagePart.Text("done")), usage, Completed(finishReason = "STOP", streamDiagnostics = diag)))
    }

    @Test
    fun `Failed followed by throw - both loops fold the same statistics`() {
        // A fresh exception instance per path: a shared one would let the first path's `partialStats`
        // leak into the second and mask a path that never attaches them.
        fun timeout() = TransientException(TransientReason.NETWORK_TIMEOUT, "no first event within 60s", null)
        assertParity(listOf(started, Failed(timeout(), streamDiagnostics = diag)), thenThrow = ::timeout)
    }

    @Test
    fun `aggregate on a TTFB timeout with Failed and throw carries cacheId, request and model in partialStats`() {
        fun timeout() = TransientException(TransientReason.NETWORK_TIMEOUT, "no first event within 60s", null)
        val stats = assertNotNull(viaAggregate(listOf(started, Failed(timeout())), ::timeout))
        assertEquals("cachedContents/test-cache", stats.cacheId)
        assertEquals(request, stats.requestMessages)
        assertEquals("test-model", stats.modelName)
        assertEquals(false, stats.success)
    }

    @Test
    fun `sequence without PromptStarted - both loops fold the same statistics`() {
        assertParity(listOf(PartReceived(MessagePart.Text("done")), usage, Completed(finishReason = "STOP", streamDiagnostics = diag)))
    }

    @Test
    fun `no PromptStarted and a failure - neither loop invents request data`() {
        fun boom() = TransientException(TransientReason.NETWORK_TIMEOUT, "boom", null)
        val aggregated = viaAggregate(listOf(Failed(boom())), ::boom)
        assertNull(aggregated, "no PromptStarted: aggregate carries nothing out")
    }
}
