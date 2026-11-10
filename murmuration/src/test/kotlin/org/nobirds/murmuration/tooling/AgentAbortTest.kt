package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.prompt.CallIds
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.assertTimeoutPreemptively
import org.junit.jupiter.api.Test
import java.time.Duration as JavaDuration
import org.nobirds.murmuration.agent.AgentExecutionEntry
import org.nobirds.murmuration.agent.PromptExecutionEntry
import org.nobirds.murmuration.agent.ExecutionContext
import org.nobirds.murmuration.agent.PipelineEventListener
import org.nobirds.murmuration.agent.RetryAgent
import org.nobirds.murmuration.agent.WrapperAgent
import org.nobirds.murmuration.agent.execute
import org.nobirds.murmuration.agent.executePipeline
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.CacheExpiredException
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Failed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.ToolResultPromptPart
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.retry.DefaultRetryStrategy
import org.nobirds.murmuration.retry.RetryPolicy
import org.nobirds.murmuration.retry.abortsAgent
import org.nobirds.murmuration.retry.causes
import org.nobirds.murmuration.retry.retryWithTimeout
import java.util.UUID
import java.io.InterruptedIOException
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds

/**
 * [AgentAbortException] leaves the agentic loop as is (never an `"Error: …"` result for the model)
 * and is terminal for every framework retry ([abortsAgent]), whatever retryable cause it carries.
 *
 * Canonical pattern: a tool throws [AgentAbortException] with a retryable `cause`; the agent runs
 * under `DefaultRetryStrategy` (round-trip retry, [RetryAgent], [retryWithTimeout]); the test counts
 * model calls / tool invocations to prove nothing was replayed.
 */
class AgentAbortTest {

    class Probe {
        var neighbourCalls = 0
        var abortCalls = 0
        var abortWith: () -> Throwable = { AgentAbortException("abort") }

        @Instrument("Neighbour tool executed before the failing one in the same round-trip.")
        fun neighbour(): String { neighbourCalls++; return "ok" }

        @Instrument("Throws whatever the test configured.")
        fun boom(): String { abortCalls++; throw abortWith() }
    }

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    private val fastPolicy = RetryPolicy(DefaultRetryStrategy, maxRetries = 3, initialBackoff = 1.milliseconds, maxBackoff = 2.milliseconds, quotaBackoffs = listOf(1.milliseconds))

    private fun batch() = listOf(
        MessagePart.ToolCall("neighbour", emptyMap(), callId = CallIds.mint()),
        MessagePart.ToolCall("boom", emptyMap(), callId = CallIds.mint()),
    )

    private fun executor(calls: AtomicInteger, captured: MutableList<Prompt> = mutableListOf()) = PromptExecutor { prompt ->
        flow {
            val n = calls.getAndIncrement()
            captured += prompt
            emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
            if (n == 0) batch().forEach { emit(PartReceived(it)) } else emit(PartReceived(MessagePart.Text("done")))
            emit(Completed(finishReason = "STOP"))
        }
    }

    private fun agent(calls: AtomicInteger, captured: MutableList<Prompt> = mutableListOf(), roundTripRetry: RetryPolicy?) =
        toolCallingAgent<Unit, Probe>(
            llm = fakeLlm, executor = executor(calls, captured), name = "AbortAgent", maxTurns = 4,
            roundTripRetry = roundTripRetry,
        ) { _, _, _ -> user { text("test prompt") } }

    private val retryableCauses: List<Pair<String, () -> Throwable>> = listOf(
        "Transient" to { TransientException(TransientReason.NETWORK_TIMEOUT, "net", null) },
        "CacheExpired" to { CacheExpiredException("c1", "expired", null) },
        "IO" to { InterruptedIOException("io") },
    )

    @Test
    fun `abort with a retryable cause is not retried by round-trip retry and keeps its cause`(): Unit = runBlocking {
        for ((name, cause) in retryableCauses) {
            val calls = AtomicInteger()
            val probe = Probe()
            val c = cause()
            probe.abortWith = { AgentAbortException("abort/$name", c) }
            val thrown = assertFailsWith<AgentAbortException>(name) {
                executePipeline { execute(agent(calls, roundTripRetry = fastPolicy), ToolCallingInput(Unit, probe)) }
            }
            // coroutines stack-trace recovery may re-wrap the exception, so look through the chain.
            assertTrue(thrown.causes.any { it === c }, "$name: cause preserved")
            assertEquals(1, calls.get(), "$name: no second round-trip")
            assertEquals(1, probe.neighbourCalls, "$name: neighbour tool executed exactly once")
            assertEquals(1, probe.abortCalls, "$name: aborting tool invoked once")
        }
    }

    @Test
    fun `abort wrapped in a foreign exception is not retried`(): Unit = runBlocking {
        val calls = AtomicInteger()
        val probe = Probe()
        // Executor-level wrapper is impossible from a tool; wrap at the RetryAgent boundary instead.
        probe.abortWith = { AgentAbortException("abort", TransientException(TransientReason.SERVER_ERROR, "5xx", null)) }
        val wrapping = object : WrapperAgent<ToolCallingInput<Unit, Probe>, Probe> {
            override val delegate = agent(calls, roundTripRetry = null)
            override val name: String get() = delegate.name
            override suspend fun ExecutionContext.execute(input: ToolCallingInput<Unit, Probe>): Probe =
                try { with(delegate) { execute(input) } } catch (e: AgentAbortException) { throw IllegalStateException("wrapped", e) }
        }
        val thrown = assertFailsWith<IllegalStateException> {
            executePipeline { execute(RetryAgent(wrapping, fastPolicy), ToolCallingInput(Unit, probe)) }
        }
        assertTrue(thrown.abortsAgent)
        assertEquals(1, calls.get())
    }

    @Test
    fun `abort with a retryable cause is not retried by RetryAgent`(): Unit = runBlocking {
        for ((name, cause) in retryableCauses) {
            val calls = AtomicInteger()
            val probe = Probe()
            probe.abortWith = { AgentAbortException("abort/$name", cause()) }
            assertFailsWith<AgentAbortException>(name) {
                executePipeline {
                    execute(RetryAgent(agent(calls, roundTripRetry = null), fastPolicy), ToolCallingInput(Unit, probe))
                }
            }
            assertEquals(1, calls.get(), "$name: whole invocation not replayed")
            assertEquals(1, probe.neighbourCalls, "$name: neighbour executed once")
        }
    }

    @Test
    fun `abort is not retried by retryWithTimeout but a plain transient failure is`(): Unit = runBlocking {
        for ((name, cause) in retryableCauses) {
            val attempts = AtomicInteger()
            assertFailsWith<AgentAbortException>(name) {
                retryWithTimeout(5.seconds, policy = fastPolicy) {
                    attempts.incrementAndGet()
                    throw AgentAbortException("abort/$name", cause())
                }
            }
            assertEquals(1, attempts.get(), "$name: single attempt")
        }
        val plain = AtomicInteger()
        val result = retryWithTimeout(5.seconds, policy = fastPolicy) {
            if (plain.incrementAndGet() < 3) throw TransientException(TransientReason.NETWORK_ERROR, "flaky", null)
            "ok"
        }
        assertEquals("ok", result)
        assertEquals(3, plain.get(), "plain transient failure still retried (regression)")
    }

    @Test
    fun `abortsAgent sees an abort anywhere in the cause chain only`() {
        assertTrue(AgentAbortException("a").abortsAgent)
        assertTrue(RuntimeException("outer", IllegalStateException("mid", AgentAbortException("a"))).abortsAgent)
        assertFalse(TransientException(TransientReason.NETWORK_ERROR, "x", null).abortsAgent)
    }

    @Test
    fun `causes cycle without an abort terminates and is not an abort`() {
        val a = RuntimeException("a")
        val b = IllegalStateException("b", a)
        a.initCause(b) // A -> B -> A
        assertTimeoutPreemptively(JavaDuration.ofSeconds(5)) {
            assertEquals(listOf(a, b), a.causes.toList())
            assertFalse(a.abortsAgent)
            assertFalse(b.abortsAgent)
        }
    }

    @Test
    fun `causes cycle with an abort terminates and sees the abort`() {
        val b = IllegalStateException("b")
        val a = AgentAbortException("a", b)
        b.initCause(a) // A(abort) -> B -> A
        assertTimeoutPreemptively(JavaDuration.ofSeconds(5)) {
            assertEquals(listOf(a, b), a.causes.toList())
            assertTrue(a.abortsAgent)
            assertTrue(b.abortsAgent)
        }
    }

    @Test
    fun `a plain tool exception is still an Error result and the model gets another round-trip`(): Unit = runBlocking {
        val calls = AtomicInteger()
        val captured = mutableListOf<Prompt>()
        val probe = Probe().apply { abortWith = { IllegalArgumentException("plain") } }
        executePipeline { execute(agent(calls, captured, roundTripRetry = null), ToolCallingInput(Unit, probe)) }
        assertTrue(calls.get() >= 2, "the model got a round-trip after the tool error")
        val results = captured[1].content.flatMap { it.parts }.filterIsInstance<ToolResultPromptPart>()
            .map { it.result.toString() }
        assertTrue(results.any { it.contains("Error: plain") }, "got $results")
    }

    /** A [Stateful] probe whose hooks throw whatever the test configured. */
    class StatefulProbe : Stateful<DefaultAgentState, StatefulProbe> {
        override val stateMachine = defaultStateMachine<StatefulProbe>()
        override val currentState: DefaultAgentState = DefaultAgentState.WORKING
        var neighbourCalls = 0
        var afterToolCallsThrows: (() -> Throwable)? = null
        var roundTripAbortedThrows: (() -> Throwable)? = null
        var roundTripAbortedCalls = 0

        @Instrument("Neighbour tool.")
        fun neighbour(): String { neighbourCalls++; return "ok" }

        @Instrument("Throws an abort.")
        fun boom(): String = throw AgentAbortException("tool-job abort")

        override suspend fun transitionTo(nextState: DefaultAgentState): TransitionResult =
            TransitionResult.Success("advanced")

        override suspend fun onAfterToolCalls(stateAtStart: DefaultAgentState) {
            afterToolCallsThrows?.let { throw it() }
        }

        override suspend fun onRoundTripAborted(state: DefaultAgentState, cause: Throwable) {
            roundTripAbortedCalls++
            roundTripAbortedThrows?.let { throw it() }
        }
    }

    /** rt0: [rt0Parts] then (optionally) a stream failure; later round-trips: plain text. */
    private fun statefulExecutor(calls: AtomicInteger, rt0Parts: List<MessagePart>, rt0Fails: Boolean) = PromptExecutor {
        flow {
            val n = calls.getAndIncrement()
            emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
            if (n == 0) rt0Parts.forEach { emit(PartReceived(it)) } else emit(PartReceived(MessagePart.Text("done")))
            if (n == 0 && rt0Fails) emit(Failed(TransientException(TransientReason.SERVER_ERROR, "stream died", null)))
            else emit(Completed(finishReason = "STOP"))
        }
    }

    private fun statefulAgent(calls: AtomicInteger, parts: List<MessagePart>, rt0Fails: Boolean = false) =
        toolCallingAgent<Unit, StatefulProbe>(
            llm = fakeLlm, executor = statefulExecutor(calls, parts, rt0Fails), name = "StatefulAbortAgent", maxTurns = 4,
            roundTripRetry = fastPolicy,
        ) { _, _, _ -> user { text("test prompt") } }

    /** Records what the pipeline was told: execution rows, retry events, failure callbacks. */
    private class AccountingRecorder : PipelineEventListener {
        val entries = mutableListOf<PromptExecutionEntry>()
        var retryEvents = 0
        var failures = 0
        override suspend fun onEntryReported(pipelineId: UUID, entry: AgentExecutionEntry) { if (entry is PromptExecutionEntry) entries += entry }
        override suspend fun onAgentRetry(pipelineId: UUID, agentName: String, attempt: Int, cause: Throwable) { retryEvents++ }
        override suspend fun onFailure(pipelineId: UUID, cause: Throwable) { failures++ }
    }

    private val wrappers: List<Pair<String, (Throwable) -> Throwable>> = listOf(
        "bare" to { a -> a },
        "wrapped" to { a -> RuntimeException("wrapped", a) },
    )

    @Test
    fun `abort from tool body, onAfterToolCalls and onRoundTripAborted leaves the loop bare or wrapped`(): Unit = runBlocking {
        val neighbourOnly = listOf<MessagePart>(MessagePart.ToolCall("neighbour", emptyMap(), callId = CallIds.mint()))
        for ((wrapName, wrap) in wrappers) {
            // Source 1: tool body (Probe.boom, after a neighbour in the same round-trip).
            run {
                val calls = AtomicInteger()
                val probe = Probe()
                val thrownAbort = wrap(AgentAbortException("abort/body"))
                probe.abortWith = { thrownAbort }
                val thrown = assertFailsWith<Throwable>("body/$wrapName") {
                    executePipeline { execute(agent(calls, roundTripRetry = fastPolicy), ToolCallingInput(Unit, probe)) }
                }
                assertTrue(thrown.causes.any { it === thrownAbort }, "body/$wrapName: same instance")
                assertEquals(1, calls.get(), "body/$wrapName: no next round-trip")
                assertEquals(1, probe.neighbourCalls, "body/$wrapName: neighbour once")
            }
            // Source 2: onAfterToolCalls.
            run {
                val calls = AtomicInteger()
                val probe = StatefulProbe()
                val thrownAbort = wrap(AgentAbortException("abort/after"))
                probe.afterToolCallsThrows = { thrownAbort }
                val thrown = assertFailsWith<Throwable>("after/$wrapName") {
                    executePipeline { execute(statefulAgent(calls, neighbourOnly), ToolCallingInput(Unit, probe)) }
                }
                assertTrue(thrown.causes.any { it === thrownAbort }, "after/$wrapName: same instance")
                assertEquals(1, calls.get(), "after/$wrapName: no next round-trip")
                assertEquals(1, probe.neighbourCalls, "after/$wrapName: neighbour once")
            }
            // Source 3: onRoundTripAborted after a retryable round-trip failure.
            run {
                val calls = AtomicInteger()
                val probe = StatefulProbe()
                val thrownAbort = wrap(AgentAbortException("abort/hook"))
                probe.roundTripAbortedThrows = { thrownAbort }
                val recorder = AccountingRecorder()
                val thrown = assertFailsWith<Throwable>("hook/$wrapName") {
                    executePipeline(pipelineEventListener = recorder) {
                        execute(statefulAgent(calls, neighbourOnly, rt0Fails = true), ToolCallingInput(Unit, probe))
                    }
                }
                // Accounting: the failed round-trip leaves exactly ONE FAILURE row (no second stub),
                // no retry event for a retry that never happened, and `onFailure` fired.
                assertEquals(1, recorder.entries.count { !it.statistics.success }, "hook/$wrapName: one FAILURE row")
                assertEquals(0, recorder.retryEvents, "hook/$wrapName: no retry event")
                assertTrue(recorder.failures >= 1, "hook/$wrapName: onFailure fired")
                assertTrue(thrown.causes.any { it === thrownAbort }, "hook/$wrapName: same instance")
                assertEquals(1, probe.roundTripAbortedCalls, "hook/$wrapName: hook called once")
                assertEquals(1, calls.get(), "hook/$wrapName: no retry after the hook's abort")
                assertEquals(1, probe.neighbourCalls, "hook/$wrapName: neighbour once")
            }
        }
    }

    @Test
    fun `a tool-job abort outranks a failed stream and is not retried`(): Unit = runBlocking {
        val calls = AtomicInteger()
        val probe = StatefulProbe()
        val thrown = assertFailsWith<Throwable> {
            executePipeline {
                execute(
                    statefulAgent(calls, listOf(MessagePart.ToolCall("boom", emptyMap(), callId = CallIds.mint())), rt0Fails = true),
                    ToolCallingInput(Unit, probe),
                )
            }
        }
        assertTrue(thrown.abortsAgent, "abort out, not the stream error: $thrown")
        assertEquals(1, calls.get(), "no retry of the round-trip")
        assertEquals(0, probe.roundTripAbortedCalls, "no abort-salvage retry path")
    }

    @Test
    fun `plain exceptions from onAfterToolCalls and onRoundTripAborted are still logged and the loop goes on`(): Unit = runBlocking {
        val neighbourOnly = listOf<MessagePart>(MessagePart.ToolCall("neighbour", emptyMap(), callId = CallIds.mint()))
        val afterCalls = AtomicInteger()
        val afterProbe = StatefulProbe().apply { afterToolCallsThrows = { IllegalStateException("plain") } }
        executePipeline { execute(statefulAgent(afterCalls, neighbourOnly), ToolCallingInput(Unit, afterProbe)) }
        assertTrue(afterCalls.get() >= 2, "loop continued after onAfterToolCalls failure")

        val hookCalls = AtomicInteger()
        val hookProbe = StatefulProbe().apply { roundTripAbortedThrows = { IllegalStateException("plain") } }
        executePipeline { execute(statefulAgent(hookCalls, neighbourOnly, rt0Fails = true), ToolCallingInput(Unit, hookProbe)) }
        assertEquals(1, hookProbe.roundTripAbortedCalls)
        assertTrue(hookCalls.get() >= 2, "retry proceeded after a plain hook failure")
    }
}
