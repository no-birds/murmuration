package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.prompt.CallIds
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonPrimitive
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
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Failed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutionEvent.UsageDelta
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.MalformedResponseException
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import java.util.Collections
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * a round-trip that fails must not close its
 * record with an empty stub built from the exception alone — tools that already ran keep their
 * outcomes (read from `ToolsExecutor`'s accumulator, which survives the throw), and whatever the
 * round-trip had assembled before breaking (`responseMessage`, tokens) rides along via
 * [org.nobirds.murmuration.prompt.PromptExecutionException.partialStats].
 *
 * Two distinct failure shapes are pinned, because they take DIFFERENT paths through the code and
 * neither subsumes the other:
 *
 * - [a stream broken mid-batch] — the fake provider emits a `Failed` EVENT; `executeTurn` catches
 *   it as `collectedFailure`, assembles the full `stats` (now moved ABOVE the throw), and attaches
 *   it via `partialStats`. This is the ONLY path with non-null `responseMessage`/tokens.
 * - [an unknown-tool call] — `ToolsExecutor.executeCall` THROWS `MalformedResponseException`
 *   directly (never via a stream event); the throw blows up `coroutineScope` from inside an
 *   `async` tool-job before `executeTurn` assembles anything at all, so `partialStats` stays
 *   null and the record's ONLY source for what happened is the accumulator.
 */
class ToolCallingAgentFailedRoundTripTest {

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    class TwoStepTool {
        val executed = Collections.synchronizedList(mutableListOf<Pair<String, String>>())

        @Instrument("Always succeeds.")
        fun stepA(@InstrumentParam("Any value") value: String): String {
            executed += "stepA" to value
            return "A:$value"
        }

        @Instrument("Always succeeds.")
        fun stepB(@InstrumentParam("Any value") value: String): String {
            executed += "stepB" to value
            return "B:$value"
        }
    }

    private class CapturingListener : PipelineEventListener {
        val entries = mutableListOf<PromptExecutionEntry>()
        val toolCallEvents = mutableListOf<Pair<UUID, String>>()

        override suspend fun onEntryReported(pipelineId: UUID, entry: AgentExecutionEntry) {
            if (entry is PromptExecutionEntry) entries += entry
        }

        override suspend fun onToolCall(pipelineId: UUID, executionId: UUID, toolName: String) {
            toolCallEvents += executionId to toolName
        }
    }

    /**
     * `PromptStarted` -> two `ToolCall` -> `UsageDelta` -> `Failed`, mid-stream: both tool jobs
     * are already dispatched by the time the stream breaks, and `executeTurn`'s `finally` awaits
     * every one of them before the failure is even inspected — so both tools genuinely run before
     * the round-trip fails.
     */
    private fun streamBreakExecutor(): PromptExecutor = PromptExecutor {
        flow {
            emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
            emit(PartReceived(MessagePart.ToolCall("stepA", mapOf("value" to JsonPrimitive("a")), callId = CallIds.mint())))
            emit(PartReceived(MessagePart.ToolCall("stepB", mapOf("value" to JsonPrimitive("b")), callId = CallIds.mint())))
            emit(
                UsageDelta(
                    input = InputTokenStatistics(total = 50, cached = 0, cacheWriteShort = 0, cacheWriteLong = 0),
                    output = OutputTokenStatistics(total = 10, thinking = 0),
                )
            )
            emit(Failed(TransientException(TransientReason.INCOMPLETE_RESPONSE, "stream dropped", null)))
        }
    }

    @Test
    fun `a stream broken mid-batch still closes its record with the tools it already ran`() = runBlocking {
        val tool = TwoStepTool()
        val listener = CapturingListener()

        assertFailsWith<TransientException> {
            executePipeline(pipelineEventListener = listener) {
                execute(
                    toolCallingAgent<Unit, TwoStepTool>(
                        llm = fakeLlm, executor = streamBreakExecutor(), name = "BreakAgent", maxTurns = 1,
                    ) { _, _, _ -> user { text("test prompt") } },
                    ToolCallingInput(Unit, tool),
                )
            }
        }

        val failure = listener.entries.single { !it.statistics.success }

        // (1) two outcomes with correct isError and FULL arguments — not a {value: ...} summary.
        assertEquals(2, failure.toolOutcomes.size, "both tools dispatched before the break must have an outcome; saw ${failure.toolOutcomes}")
        val byName = failure.toolOutcomes.associateBy { it.name }
        assertEquals(mapOf("value" to JsonPrimitive("a")), byName.getValue("stepA").args)
        assertEquals(mapOf("value" to JsonPrimitive("b")), byName.getValue("stepB").args)
        assertTrue(byName.values.all { !it.isError }, "both stepA/stepB succeeded, neither is an error")
        assertTrue(byName.values.all { it.exit == ToolOutcome.Exit.INVOKED })

        // (2) non-empty responseMessage carrying BOTH tool_call parts.
        val response = assertNotNull(failure.statistics.responseMessage, "an obryv-stream record must keep its responseMessage")
        assertEquals(
            listOf("stepA", "stepB"),
            response.parts.filterIsInstance<MessagePart.ToolCall>().map { it.name },
        )

        // (3) non-zero partial tokens — the UsageDelta burned before the break must survive.
        assertEquals(60L, failure.statistics.tokens.totalTokens)

        // (4) non-empty modelName.
        assertEquals("test-model", failure.statistics.modelName)

        // (5) S-A: onToolCall events for this executionId == outcomes with exit INVOKED.
        val eventsThisRt = listener.toolCallEvents.filter { it.first == failure.executionId }.map { it.second }
        assertEquals(listOf("stepA", "stepB").sorted(), eventsThisRt.sorted())
    }

    class SpyThreeStepTool {
        val executed = Collections.synchronizedList(mutableListOf<String>())

        @Instrument("Always succeeds.")
        fun stepA(@InstrumentParam("Any value") value: String): String {
            executed += "stepA"
            return "A:$value"
        }

        @Instrument("Always succeeds.")
        fun stepB(@InstrumentParam("Any value") value: String): String {
            executed += "stepB"
            return "B:$value"
        }

        @Instrument("Always succeeds.")
        fun stepC(@InstrumentParam("Any value") value: String): String {
            executed += "stepC"
            return "C:$value"
        }
    }

    /**
     * Four calls in one batch — `stepA`, `stepB`, a tool the tool class does NOT declare, `stepC`
     * — issued through the (default) [SequentialToolCallScheduler]. How many of stepA/stepB/stepC
     * actually run before `coroutineScope` is torn down by the `async` job that throws on the
     * unknown-tool call is NOT deterministic (the scheduler's
     * queue does not necessarily drain before cancellation lands) — the assertions below do not
     * depend on how many, only on the accumulator agreeing EXACTLY with what the tool itself
     * recorded as executed.
     */
    private fun unknownToolExecutor(): PromptExecutor = PromptExecutor {
        flow {
            emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
            emit(PartReceived(MessagePart.ToolCall("stepA", mapOf("value" to JsonPrimitive("a")), callId = CallIds.mint())))
            emit(PartReceived(MessagePart.ToolCall("stepB", mapOf("value" to JsonPrimitive("b")), callId = CallIds.mint())))
            emit(PartReceived(MessagePart.ToolCall("doesNotExist", mapOf("value" to JsonPrimitive("x")), callId = CallIds.mint())))
            emit(PartReceived(MessagePart.ToolCall("stepC", mapOf("value" to JsonPrimitive("c")), callId = CallIds.mint())))
            emit(Completed(finishReason = "STOP"))
        }
    }

    @Test
    fun `an unknown-tool call closes its record with exactly the outcomes the accumulator actually saw run`() = runBlocking {
        val tool = SpyThreeStepTool()
        val listener = CapturingListener()

        val thrown = assertFailsWith<MalformedResponseException> {
            executePipeline(pipelineEventListener = listener) {
                execute(
                    toolCallingAgent<Unit, SpyThreeStepTool>(
                        llm = fakeLlm, executor = unknownToolExecutor(), name = "UnknownToolAgent", maxTurns = 1,
                    ) { _, _, _ -> user { text("test prompt") } },
                    ToolCallingInput(Unit, tool),
                )
            }
        }
        assertTrue(
            thrown.message.orEmpty().contains("Unknown tool: doesNotExist"),
            "the unresolved tool's name comes from errorMessage, not from an outcome; saw: ${thrown.message}",
        )

        val failure = listener.entries.single { !it.statistics.success }

        assertTrue(thrown.message.orEmpty() == failure.statistics.errorMessage, "the record's errorMessage must carry the same 'Unknown tool: …' text")

        // responseMessage is null on this path: `executeTurn` never reaches its own assembly —
        // the throw blows up coroutineScope from inside a tool-job, before that code runs.
        assertNull(failure.statistics.responseMessage, "the unknown-tool path never assembles a responseMessage")
        assertEquals(0L, failure.statistics.tokens.totalTokens, "no UsageDelta arrived on this path")

        // Bidirectional spy every tool the SPY recorded as executed has a matching
        // INVOKED outcome in the record, and vice versa — regardless of how many of the three
        // actually got to run before cancellation cut the queue off.
        val invokedNames = failure.toolOutcomes.filter { it.exit == ToolOutcome.Exit.INVOKED }.map { it.name }.sorted()
        val spiedNames = tool.executed.toList().sorted()
        assertEquals(spiedNames, invokedNames, "accumulator must agree EXACTLY with what the instrument spy recorded as executed")
        assertTrue("doesNotExist" !in invokedNames, "the call that threw never produces an outcome at all — it never reached any of executeCall's four returns")
    }
}
