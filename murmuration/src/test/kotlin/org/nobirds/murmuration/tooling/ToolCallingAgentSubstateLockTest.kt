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
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.ToolResultPromptPart
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * [Stateful.availableTools] is the framework's only DYNAMIC tool gate: on top of the static
 * `@Phases` gate, a builder may collapse the allowlist further given its live substate. This
 * pins that the gate is both ENFORCED (a narrowed-out call returns an error and never invokes
 * the tool) and ADVERTISED (the rendered "Tools available" block drops the locked tool), using
 * the real [ToolCallingAgent] with a fake executor.
 *
 * Models the TaskAwareBuilder oscillation fix: once the builder enters its "only a selfCheck is
 * owed" substate, a working mutator (`work`) is locked so a reflexive re-issue can't spawn a
 * duplicate — only the finishing tool survives.
 */
class ToolCallingAgentSubstateLockTest {

    /**
     * A [Stateful] that enters a locked substate after the first `work` call: from then on
     * `availableTools` drops `work`, leaving only the always-available `transitionTo`.
     */
    class LockTool : Stateful<DefaultAgentState, LockTool> {
        override val stateMachine = defaultStateMachine<LockTool>()
        private var state = DefaultAgentState.WORKING
        override val currentState: DefaultAgentState get() = state

        var workInvocations = 0
            private set
        private var locked = false

        override fun availableTools(state: DefaultAgentState, staticallyAllowed: Set<String>): Set<String> =
            if (locked) staticallyAllowed - "work" else staticallyAllowed

        @Instrument("Do a unit of work — after the first call the tool enters a locked substate.")
        fun work(@InstrumentParam("Any value") value: String): String {
            workInvocations++
            locked = true
            return "worked"
        }

        override suspend fun transitionTo(nextState: DefaultAgentState): TransitionResult =
            TransitionResult.Success("ok").also { state = nextState }
    }

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    /** rt1 & rt2 both call `work("x")`; rt3 emits plain text so the loop winds down at maxTurns. */
    private fun executor(captured: MutableList<Prompt>): PromptExecutor = PromptExecutor { prompt ->
        flow {
            val turn = captured.size
            captured += prompt
            emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
            if (turn < 2) {
                emit(PartReceived(MessagePart.ToolCall("work", mapOf<String, JsonElement>("value" to JsonPrimitive("x")), callId = CallIds.mint())))
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

    /**
     * The CURRENT round-trip's "Tools available in phase ..." block. The raw captured history
     * accumulates one such block per round-trip (the cross-round-trip pruning runs in a
     * send-time decorator, not in this pre-executor prompt), so the LAST match is this
     * round-trip's — see [ToolCallingAgent.appendPhaseDescription].
     */
    private fun Prompt.phaseToolsBlock(): String =
        content.flatMap { it.parts }
            .filterIsInstance<TextPromptPart>()
            .map { it.text }
            .last { it.contains("Tools available in phase") }

    @Test
    fun `a locked tool is dropped from the phase block and rejected at the call gate`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = LockTool()
        val agent = toolCallingAgent<Unit, LockTool>(
            llm = fakeLlm, executor = executor(captured), name = "LockAgent", maxTurns = 3,
        ) { _, _, _ -> user { text("test prompt") } }
        executePipeline { execute(agent, ToolCallingInput(Unit, tool)) }

        assertEquals(3, captured.size, "three round-trips ran")

        // rt1 ran before the lock: `work` is advertised and executes.
        assertTrue("work" in captured[0].phaseToolsBlock(), "rt1 must advertise the unlocked tool")
        // rt2 is rendered AFTER the lock engaged (work ran in rt1): the block drops `work`.
        assertFalse("work" in captured[1].phaseToolsBlock(), "rt2 must not advertise the locked tool; saw: ${captured[1].phaseToolsBlock()}")
        assertTrue("transitionTo" in captured[1].phaseToolsBlock(), "the always-available control gate stays")

        // The gate is enforced, not just advertised: rt2's `work` call never invoked the tool.
        assertEquals(1, tool.workInvocations, "the locked repeat must be rejected before invoking the tool")
        // rt3's request carries rt2's rejected result (an Error, not a NO-OP short-circuit).
        val rt3Results = captured[2].toolResultTexts()
        assertTrue(
            rt3Results.any { it.startsWith("Error:") && it.contains("locked") },
            "the locked call must return an explicit lock error; saw: $rt3Results",
        )
    }
}
