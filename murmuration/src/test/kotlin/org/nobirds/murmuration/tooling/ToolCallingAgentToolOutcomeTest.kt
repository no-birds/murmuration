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
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * `ToolsExecutor.executeCall` has FOUR distinct returns, and
 * "counted" means every one of them — not just the happy path — produces a [ToolOutcome]. This
 * pins that with ONE round-trip batch that deliberately drives all four exits at once, plus the
 * event half of the C-A reconciliation: `onToolCall` fires (sharing this round-trip's `executionId`)
 * for the two exits whose dispatch survives to `ec.afterToolCall`, and stays silent for the two
 * early refusals — see [ToolCallingAgent] `:1193`/`:1221` (dispatch sites) against the four
 * `executeCall` returns.
 */
class ToolCallingAgentToolOutcomeTest {

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    /**
     * One instrument per `executeCall` exit: [phaseOnly] is never in-phase (exit PHASE_REFUSED),
     * [locked] is narrowed out by [availableTools] (exit LOCKED), [repeatable] is idempotent so a
     * third identical issue reaches the hard-skip (exit REPEAT_SKIPPED), [errorPrefixed] and
     * `transitionTo` both reach exit INVOKED — one classified an error via [ErrorMatcher]'s prefix
     * convention, the other HONESTLY via [TransitionResult.success] (see `executeCall`'s comment on
     * why these are two different classification paths feeding the same `isError`).
     */
    class FourExitsTool : Stateful<DefaultAgentState, FourExitsTool> {
        override val stateMachine = defaultStateMachine<FourExitsTool>()
        override val currentState: DefaultAgentState = DefaultAgentState.WORKING

        var repeatableCalls = 0
            private set
        var lockedCalls = 0
            private set

        override fun availableTools(state: DefaultAgentState, staticallyAllowed: Set<String>): Set<String> =
            staticallyAllowed - "locked"

        @Phases("NEVER")
        @Instrument("Never in-phase in this run — always PHASE_REFUSED.")
        fun phaseOnly(@InstrumentParam("Any value") value: String): String = "unreachable"

        @Instrument("Locked by dynamic substate narrowing — always LOCKED.")
        fun locked(@InstrumentParam("Any value") value: String): String {
            lockedCalls++
            return "should not run"
        }

        @Instrument("Idempotent call, issued 3x to reach the hard-skip.")
        fun repeatable(@InstrumentParam("Any value") value: String): String {
            repeatableCalls++
            return "stable"
        }

        @Instrument("Returns an ErrorMatcher-recognised error result.")
        fun errorPrefixed(@InstrumentParam("Any value") value: String): String = "ERROR: boom"

        // Always refuses — the honest TransitionResult.success classification, not the
        // ErrorMatcher prefix heuristic, is what must flag this outcome as an error.
        override suspend fun transitionTo(nextState: DefaultAgentState): TransitionResult =
            TransitionResult.Failure("rejected")
    }

    private val batch0 = listOf(
        MessagePart.ToolCall("phaseOnly", mapOf("value" to JsonPrimitive("p")), callId = CallIds.mint()),
        MessagePart.ToolCall("locked", mapOf("value" to JsonPrimitive("l")), callId = CallIds.mint()),
        MessagePart.ToolCall("repeatable", mapOf("value" to JsonPrimitive("r")), callId = CallIds.mint()),
        MessagePart.ToolCall("repeatable", mapOf("value" to JsonPrimitive("r")), callId = CallIds.mint()),
        MessagePart.ToolCall("repeatable", mapOf("value" to JsonPrimitive("r")), callId = CallIds.mint()),
        MessagePart.ToolCall("errorPrefixed", mapOf("value" to JsonPrimitive("e")), callId = CallIds.mint()),
        MessagePart.ToolCall("transitionTo", mapOf("nextState" to JsonPrimitive("FINISHED")), callId = CallIds.mint()),
    )

    /** rt0 issues [batch0]; every later round-trip emits plain text so the loop winds down. */
    private fun executor(captured: MutableList<Prompt>): PromptExecutor = PromptExecutor { prompt ->
        flow {
            val turn = captured.size
            captured += prompt
            emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
            (if (turn == 0) batch0 else listOf(MessagePart.Text("done"))).forEach { emit(PartReceived(it)) }
            emit(Completed(finishReason = "STOP"))
        }
    }

    /** Captures both halves of the C-A reconciliation: reported entries (with [PromptExecutionEntry.toolOutcomes]) and `onToolCall` events. */
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

    @Test
    fun `executeCall's four returns each produce a ToolOutcome, and onToolCall fires only for two of them`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = FourExitsTool()
        val listener = CapturingListener()

        executePipeline(pipelineEventListener = listener) {
            execute(
                toolCallingAgent<Unit, FourExitsTool>(
                    llm = fakeLlm, executor = executor(captured), name = "FourExitsAgent", maxTurns = 2,
                ) { _, _, _ -> user { text("test prompt") } },
                ToolCallingInput(Unit, tool),
            )
        }

        val rt0 = listener.entries.first { it.toolOutcomes.isNotEmpty() }
        assertEquals(
            7, rt0.toolOutcomes.size,
            "every one of the batch's 7 calls must have produced an outcome; saw ${rt0.toolOutcomes.map { it.name to it.exit }}",
        )

        val byName = rt0.toolOutcomes.groupBy { it.name }

        val phaseOnly = byName.getValue("phaseOnly").single()
        assertEquals(ToolOutcome.Exit.PHASE_REFUSED, phaseOnly.exit)
        assertFalse(phaseOnly.invoked)
        assertTrue(phaseOnly.isError)
        assertEquals(mapOf("value" to JsonPrimitive("p")), phaseOnly.args, "the FULL call args must be carried, not a key summary")

        val locked = byName.getValue("locked").single()
        assertEquals(ToolOutcome.Exit.LOCKED, locked.exit)
        assertFalse(locked.invoked)
        assertTrue(locked.isError)
        assertEquals(mapOf("value" to JsonPrimitive("l")), locked.args)
        assertEquals(0, tool.lockedCalls, "a LOCKED exit must never reach the instrument")

        val repeats = byName.getValue("repeatable").sortedBy { it.ordinal }
        assertEquals(3, repeats.size)
        assertEquals(
            listOf(ToolOutcome.Exit.INVOKED, ToolOutcome.Exit.INVOKED, ToolOutcome.Exit.REPEAT_SKIPPED),
            repeats.map { it.exit },
        )
        assertFalse(repeats.last().invoked, "the third identical call must be hard-skipped, not invoked")
        assertFalse(repeats.last().isError, "skipResultFor only ever replays a NON-error stored result")
        assertEquals(2, tool.repeatableCalls, "the tool itself ran only twice — the third call was short-circuited")

        val errorPrefixed = byName.getValue("errorPrefixed").single()
        assertEquals(ToolOutcome.Exit.INVOKED, errorPrefixed.exit)
        assertTrue(errorPrefixed.invoked)
        assertTrue(errorPrefixed.isError, "the ErrorMatcher must classify the ERROR-prefixed result as an error")

        val transitionCall = byName.getValue("transitionTo").single()
        assertEquals(ToolOutcome.Exit.INVOKED, transitionCall.exit)
        assertTrue(transitionCall.invoked)
        assertTrue(transitionCall.isError, "a TransitionResult.Failure must be classified as an error HONESTLY, not via prefix-matching")

        // onToolCall dispatches from ec.afterToolCall at TWO of executeCall's returns
        // (hard-skip `:1194` and invoked `:1265`) — not at the two early refusals (`:1159`/`:1174`).
        // It must share this round-trip's executionId, the same correlation onAgentTurn carries.
        val eventsThisRt = listener.toolCallEvents.filter { it.first == rt0.executionId }.map { it.second }
        assertEquals(
            listOf("repeatable", "repeatable", "repeatable", "errorPrefixed", "transitionTo").sorted(),
            eventsThisRt.sorted(),
            "onToolCall must fire for every REPEAT_SKIPPED/INVOKED outcome, correlated by this round-trip's executionId",
        )
        assertTrue("phaseOnly" !in eventsThisRt, "a PHASE_REFUSED exit must not dispatch onToolCall")
        assertTrue("locked" !in eventsThisRt, "a LOCKED exit must not dispatch onToolCall")
    }
}
