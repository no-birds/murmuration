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
import org.nobirds.murmuration.prompt.InputTokenStatistics
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.OutputTokenStatistics
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Failed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutionEvent.UsageDelta
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.QuotaExceededException
import org.nobirds.murmuration.prompt.QuotaKind
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.retry.DefaultRetryStrategy
import org.nobirds.murmuration.retry.RetryPolicy
import java.util.UUID
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds

/**
 * Per-round-trip recovery of [ToolCallingAgent] (`roundTripRetry`): a retryable failure of a
 * single round-trip is retried inside the tool loop on the preserved conversation instead of
 * aborting the whole invocation (which the outer RetryAgent would replay from scratch).
 */
class ToolCallingAgentRoundTripRetryTest {

    class PingTool {
        @Instrument("Ping the test tool.")
        fun ping(): String = "pong"
    }

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    private val fastRetryPolicy = RetryPolicy(
        strategy = DefaultRetryStrategy,
        maxRetries = 2,
        initialBackoff = 1.milliseconds,
        maxBackoff = 2.milliseconds,
        quotaBackoffs = listOf(1.milliseconds),
    )

    /** Emits a broken stream for the first [failures] calls, then a clean completion. */
    private fun executor(failures: Int, calls: AtomicInteger) = PromptExecutor { _ ->
        flow {
            val n = calls.incrementAndGet()
            emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
            if (n <= failures) {
                emit(Failed(TransientException(TransientReason.NETWORK_TIMEOUT, "stream broke", null)))
            } else {
                emit(PartReceived(MessagePart.Text("done")))
                emit(Completed(finishReason = "STOP"))
            }
        }
    }

    private class RetryRecorder : PipelineEventListener {
        val attempts = mutableListOf<Int>()
        override suspend fun onAgentRetry(pipelineId: UUID, agentName: String, attempt: Int, cause: Throwable) {
            attempts += attempt
        }
    }

    /** Captures RUNNING-row starts and reported entries so we can assert the row lifecycle. */
    private class RowRecorder : PipelineEventListener {
        val turnIds = mutableListOf<UUID>()
        val entries = mutableListOf<PromptExecutionEntry>()
        override suspend fun onAgentTurn(pipelineId: UUID, executionId: UUID, agentName: String) {
            turnIds += executionId
        }
        override suspend fun onEntryReported(pipelineId: UUID, entry: AgentExecutionEntry) {
            if (entry is PromptExecutionEntry) entries += entry
        }
    }

    /**
     * Emits a 429 (mapped [QuotaExceededException], as the Azure mapper produces) for the first
     * [failures] calls, then a clean completion. `retryAfter` models the provider's `Retry-After`
     * header so the retry path honours it.
     */
    private fun quotaExecutor(failures: Int, calls: AtomicInteger, retryAfter: Duration? = null) = PromptExecutor { _ ->
        flow {
            val n = calls.incrementAndGet()
            emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
            if (n <= failures) {
                emit(Failed(QuotaExceededException(QuotaKind.RATE_LIMIT, retryAfter, "429: rate limit exceeded", null)))
            } else {
                emit(PartReceived(MessagePart.Text("done")))
                emit(Completed(finishReason = "STOP"))
            }
        }
    }

    /** Emits partial usage, then breaks — models a stream that burned tokens before dropping. */
    private fun failingWithUsageExecutor(calls: AtomicInteger) = PromptExecutor { _ ->
        flow {
            calls.incrementAndGet()
            emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
            emit(UsageDelta(input = InputTokenStatistics(total = 100, cached = 0, cacheWriteShort = 0, cacheWriteLong = 0), output = OutputTokenStatistics(total = 20, thinking = 15)))
            emit(Failed(TransientException(TransientReason.INCOMPLETE_RESPONSE, "stream broke after thinking", null)))
        }
    }

    @Test
    fun `failed round-trip closes its RUNNING row via a FAILURE entry keyed by executionId`(): Unit = runBlocking {
        val calls = AtomicInteger()
        val rows = RowRecorder()
        assertFailsWith<TransientException> {
            executePipeline(pipelineEventListener = rows) {
                // No round-trip retry: the single attempt fails and propagates.
                val agent = toolCallingAgent<Unit, PingTool>(
                    llm = fakeLlm, executor = failingWithUsageExecutor(calls), name = "RowProbeAgent",
                    maxTurns = 4, roundTripRetry = null,
                ) { _, _, _ -> user { text("test prompt") } }
                execute(agent, ToolCallingInput(Unit, PingTool()))
            }
        }

        // Exactly one RUNNING row opened, and exactly one FAILURE entry closing it (no duplicate
        // from the outer agent catch).
        assertEquals(1, rows.turnIds.size, "one round-trip started one RUNNING row")
        assertEquals(1, rows.entries.size, "exactly one entry reported — no duplicate FAILURE row")
        val entry = rows.entries.single()
        assertEquals(rows.turnIds.single(), entry.executionId, "entry must UPDATE the RUNNING row (same executionId)")
        assertTrue(!entry.statistics.success, "the entry is a FAILURE")
        assertEquals("INCOMPLETE_RESPONSE", entry.statistics.errorCode)
        // model_name must be filled even on failure (falls back to the configured Llm.model) —
        // LlmCostCalculator groups by modelName and skips empty ones, so an unfilled value would
        // silently drop this failure's partial tokens from cost accounting and admin's byModel
        // breakdown.
        assertEquals("test-model", entry.statistics.modelName)
        assertEquals("test-model", entry.statistics.requestedModelName)
        // Partial tokens burned before the break are carried into the FAILURE row, not zeroed.
        assertEquals(100L, entry.statistics.tokens.input.total)
        assertEquals(20L, entry.statistics.tokens.output.total)
        assertEquals(15L, entry.statistics.tokens.output.thinking)
    }

    @Test
    fun `retried attempts each land as their own FAILURE row`(): Unit = runBlocking {
        val calls = AtomicInteger()
        val rows = RowRecorder()
        assertFailsWith<TransientException> {
            executePipeline(pipelineEventListener = rows) {
                // Always failing, retry budget 2 → 3 attempts total, each its own RUNNING row.
                execute(agent(calls, failures = Int.MAX_VALUE, policy = fastRetryPolicy), ToolCallingInput(Unit, PingTool()))
            }
        }
        assertEquals(3, calls.get(), "initial attempt plus two retries")
        assertEquals(3, rows.turnIds.size, "each attempt opens its own RUNNING row")
        assertEquals(3, rows.entries.size, "each attempt closes its row — retries visible as FAILURE rows")
        assertEquals(rows.turnIds.toSet(), rows.entries.map { it.executionId }.toSet(), "entries match their RUNNING rows")
        assertTrue(rows.entries.all { !it.statistics.success }, "all three are FAILURE rows")
        assertTrue(rows.entries.all { it.statistics.modelName == "test-model" }, "every retried FAILURE row still carries model_name")
    }

    private fun agent(calls: AtomicInteger, failures: Int, policy: RetryPolicy?, maxTurns: Int = 4) =
        toolCallingAgent<Unit, PingTool>(
            llm = fakeLlm,
            executor = executor(failures, calls),
            name = "RetryProbeAgent",
            maxTurns = maxTurns,
            roundTripRetry = policy,
        ) { _, _, _ -> user { text("test prompt") } }

    @Test
    fun `retryable round-trip failure recovers inside the loop`() = runBlocking {
        val calls = AtomicInteger()
        val recorder = RetryRecorder()
        executePipeline(pipelineEventListener = recorder) {
            // maxTurns=2: a retry re-attempts the SAME productive turn and does NOT consume the
            // productive-turn budget, so a recovered run spends 2 productive turns (the default
            // state machine has no terminal transition in this probe) plus 1 retry = 3 calls.
            execute(agent(calls, failures = 1, policy = fastRetryPolicy, maxTurns = 2), ToolCallingInput(Unit, PingTool()))
        }
        // Attempt 1 broke and was retried, then 2 productive turns completed: no exception escaped.
        assertEquals(3, calls.get(), "one retry plus two productive turns")
        assertEquals(listOf(1), recorder.attempts, "exactly one retry must be reported")
    }

    @Test
    fun `retryable failure on the last productive turn is not swallowed`() = runBlocking {
        // Regression for the MapBattle#15 429: the retryable failure lands on the FINAL productive
        // turn (maxTurns=1). Before the fix the scheduled retry had no loop pass left to run in, so
        // it was silently dropped and the invocation returned partial "maxTurns exhausted" work —
        // neither retried in-loop nor propagated to the outer RetryAgent. Now the retry re-attempts
        // the same turn regardless of position: the run recovers with no exception.
        val calls = AtomicInteger()
        val recorder = RetryRecorder()
        executePipeline(pipelineEventListener = recorder) {
            execute(agent(calls, failures = 1, policy = fastRetryPolicy, maxTurns = 1), ToolCallingInput(Unit, PingTool()))
        }
        assertEquals(2, calls.get(), "the failed last-turn round-trip must be re-prompted once")
        assertEquals(listOf(1), recorder.attempts, "the retry must actually fire on the last turn")
    }

    @Test
    fun `429 on the last productive turn retries with backoff and recovers`() = runBlocking {
        // The exact reported scenario: a 429 (QuotaExceededException, RATE_LIMIT) on the final
        // productive turn, carrying a Retry-After. The turn must retry in-loop (honouring the
        // hint) and recover — not die with an unfinished task.
        val calls = AtomicInteger()
        val recorder = RetryRecorder()
        val agent = toolCallingAgent<Unit, PingTool>(
            llm = fakeLlm,
            executor = quotaExecutor(failures = 1, calls = calls, retryAfter = 1.milliseconds),
            name = "QuotaProbeAgent",
            maxTurns = 1,
            roundTripRetry = fastRetryPolicy,
        ) { _, _, _ -> user { text("test prompt") } }
        executePipeline(pipelineEventListener = recorder) {
            execute(agent, ToolCallingInput(Unit, PingTool()))
        }
        assertEquals(2, calls.get(), "the 429 last-turn round-trip must be retried once and recover")
        assertEquals(listOf(1), recorder.attempts, "exactly one retry must be reported for the 429")
    }

    @Test
    fun `consecutive budget exhausted - failure propagates`(): Unit = runBlocking {
        val calls = AtomicInteger()
        assertFailsWith<TransientException> {
            executePipeline {
                // Always failing: maxRetries=2 → attempts 1 (initial) + 2 retries, then throw.
                execute(agent(calls, failures = Int.MAX_VALUE, policy = fastRetryPolicy), ToolCallingInput(Unit, PingTool()))
            }
        }
        assertEquals(3, calls.get(), "initial attempt plus two retries")
    }

    @Test
    fun `recovery off by default - failure propagates immediately`(): Unit = runBlocking {
        val calls = AtomicInteger()
        assertFailsWith<TransientException> {
            executePipeline {
                execute(agent(calls, failures = 1, policy = null), ToolCallingInput(Unit, PingTool()))
            }
        }
        assertEquals(1, calls.get(), "no retry without a policy")
    }
}
