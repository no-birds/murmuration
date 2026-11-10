package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.prompt.CallIds
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.agent.AgentEventListener
import org.nobirds.murmuration.agent.execute
import org.nobirds.murmuration.agent.executePipeline
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Regression coverage for the legacy [executePipeline] entry point's global call-limit check
 * (`AgentScope.kt`, [org.nobirds.murmuration.agent.executePipeline]). The strategy/message
 * ("Pipeline reached global limit of N LLM calls") is stated in terms of LLM round-trips, and
 * `ExecutionContext.beforeCall()` is invoked at exactly that cadence — once per
 * `promptExecutor.execute(...)` — but the wrapper used to compare the configured [maxCalls]
 * against [org.nobirds.murmuration.agent.ExecutionMetrics.toolCalls] (the count of individual
 * tool invocations across all round-trips) instead of
 * [org.nobirds.murmuration.agent.ExecutionMetrics.requests] (the count of round-trips itself).
 * A model that batches many tool-calls into a single response inflates `toolCalls` far faster
 * than `requests`, which used to trip the "LLM calls" budget after only a handful of real LLM
 * round-trips.
 */
class ExecutePipelineCallLimitTest {

    class PingTool {
        @Instrument("Ping the test tool.")
        fun ping(): String = "pong"
    }

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    /**
     * Emits [toolCallsPerTurn] batched tool-calls in a single response on every round-trip —
     * models a provider that packs a lot of "real work" into one LLM call.
     */
    private fun batchingExecutor(toolCallsPerTurn: Int, requestCount: AtomicInteger) = PromptExecutor { _ ->
        flow {
            requestCount.incrementAndGet()
            emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
            repeat(toolCallsPerTurn) {
                emit(PartReceived(MessagePart.ToolCall(name = "ping", args = emptyMap(), callId = CallIds.mint())))
            }
            emit(Completed(finishReason = "STOP"))
        }
    }

    private class CallLimitRecorder : AgentEventListener, ToolCallingListener {
        val reasons = mutableListOf<String>()
        override suspend fun onCallLimitExhausted(state: AgentState, reason: String) {
            reasons += reason
        }
    }

    private fun agent(executor: PromptExecutor, maxTurns: Int) =
        toolCallingAgent<Unit, PingTool>(
            llm = fakeLlm,
            executor = executor,
            name = "BudgetProbeAgent",
            maxTurns = maxTurns,
            roundTripRetry = null,
        ) { _, _, _ -> user { text("test prompt") } }

    @Test
    fun `heavy tool-call batching per round-trip does not exhaust the LLM-call budget`(): Unit = runBlocking {
        val requestCount = AtomicInteger()
        val recorder = CallLimitRecorder()
        // 5 round-trips allowed by the agent's own maxTurns; the pipeline-wide budget (10
        // "LLM calls") is generous relative to that. Each round-trip batches 60 tool-calls —
        // 300 tool-calls total, 30x the configured maxCalls. If the limit were (still)
        // compared against the tool-call counter instead of the request counter, the run
        // would be cut short well before the 5th round-trip.
        executePipeline(maxCalls = 10) {
            register(recorder)
            execute(agent(batchingExecutor(toolCallsPerTurn = 60, requestCount = requestCount), maxTurns = 5), ToolCallingInput(Unit, PingTool()))
        }
        assertEquals(5, requestCount.get(), "all 5 round-trips must run — tool-call volume must not count against the LLM-call budget")
        assertTrue(recorder.reasons.isEmpty(), "budget must not have been reported exhausted: ${recorder.reasons}")
    }

    @Test
    fun `exceeding the actual LLM round-trip count stops the pipeline gracefully`(): Unit = runBlocking {
        val requestCount = AtomicInteger()
        val recorder = CallLimitRecorder()
        // No tool-calls at all this time — only the round-trip count can trip the budget.
        // maxTurns=10 gives the agent plenty of local headroom; maxCalls=3 is the actual
        // constraint under test.
        executePipeline(maxCalls = 3) {
            register(recorder)
            execute(agent(batchingExecutor(toolCallsPerTurn = 0, requestCount = requestCount), maxTurns = 10), ToolCallingInput(Unit, PingTool()))
        }
        // DefaultCallLimitStrategy stops once requests > maxCalls: round-trips 1..4 run
        // (beforeCall sees 0,1,2,3 requests so far), the 5th is refused.
        assertEquals(4, requestCount.get(), "must stop right after crossing the configured LLM-call budget")
        assertEquals(1, recorder.reasons.size, "call-limit-exhausted must fire exactly once")
        assertTrue(
            recorder.reasons.single().contains("LLM calls"),
            "message must describe the LLM-call budget it actually enforces: ${recorder.reasons.single()}",
        )
    }
}
