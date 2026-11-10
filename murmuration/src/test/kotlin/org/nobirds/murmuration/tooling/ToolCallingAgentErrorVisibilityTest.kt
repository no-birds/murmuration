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
import kotlin.test.assertTrue

/**
 * The forced error-visibility round-trip (see [ToolCallingAgent.errorMatcher]) closes the blind spot
 * where a batch that both hides an error result AND moves the phase on (advancing it, or finishing
 * the invocation) on the SAME round-trip swallows that error — the model never reacts to it. When the
 * matcher flags a result of such a batch, the framework HOLDS the between-round-trips transition
 * apply for one forced round-trip so the model sees the error (already in the conversation) and can
 * fix it in the current phase before the state machine is allowed to move.
 *
 * The hold budget is per "place" (source phase + the transition the batch requested): never twice in
 * a row at the same place — a repeat is applied with a WARN so an endlessly-erroring model cannot
 * livelock — while an errored batch requesting no framework-visible transition keeps the original
 * budget of one hold per invocation. The first group of tests below covers the finishing/immediate
 * cases and that placeless budget; the second (`DeferredFlowTool`) covers MID-flow transitions and
 * the per-place budget.
 *
 * The guard is a FRAMEWORK DEFAULT, not an opt-in: the agents below are built with no
 * error-classification argument at all, and the hold still fires off the resolved [ErrorMatcher]'s
 * prefix. That is the whole point of these tests — an agent must not have to remember to ask.
 *
 * They drive the real [ToolCallingAgent] with a fake executor and a DEFERRED-applier [Stateful]
 * (transitions land in [Stateful.onAfterToolCalls], as the pipeline builders do), which is the shape
 * the hold protects: skipping that apply keeps the phase from advancing.
 */
class ToolCallingAgentErrorVisibilityTest {

    /**
     * A deferred-applier [Stateful] over the default WORKING → FINISHED machine. `finish` only
     * ARMS the terminal transition; it is applied at the end of the round-trip in [onAfterToolCalls]
     * (skipped by the framework's forced-error hold, so `finish` alone does not finish a batch that
     * also errored). `doError` returns an `"ERROR:"`-prefixed result; `fix` is a clean corrective.
     */
    class DeferredFinishTool(private val errorText: String = "ERROR: boom") : Stateful<DefaultAgentState, DeferredFinishTool> {
        override val stateMachine = defaultStateMachine<DeferredFinishTool>()
        private var state = DefaultAgentState.WORKING
        override val currentState: DefaultAgentState get() = state

        private var finishArmed = false
        var errorCalls = 0
            private set
        var fixCalls = 0
            private set

        @Instrument("Returns an error result (simulates a rejected mutation hidden inside a batch).")
        fun doError(@InstrumentParam("Any value") value: String): String {
            errorCalls++
            return errorText
        }

        @Instrument("A clean corrective call.")
        fun fix(@InstrumentParam("Any value") value: String): String {
            fixCalls++
            return "fixed: $value"
        }

        @Instrument("Arm finishing at the end of this response.")
        fun finish(@InstrumentParam("Reason") reason: String): String {
            finishArmed = true
            return "finish requested"
        }

        override suspend fun onRoundTripStart(state: DefaultAgentState) {
            finishArmed = false
        }

        override suspend fun onAfterToolCalls(stateAtStart: DefaultAgentState) {
            if (finishArmed) state = DefaultAgentState.FINISHED
        }

        // Never invoked in these tests (finishing goes through `finish` + onAfterToolCalls).
        override suspend fun transitionTo(nextState: DefaultAgentState): TransitionResult =
            TransitionResult.Success("ok")
    }

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    private fun errorBatch(): List<MessagePart> = listOf(
        MessagePart.ToolCall("doError", mapOf<String, JsonElement>("value" to JsonPrimitive("x")), callId = CallIds.mint()),
        MessagePart.ToolCall("finish", mapOf<String, JsonElement>("reason" to JsonPrimitive("done")), callId = CallIds.mint()),
    )

    private fun cleanBatch(): List<MessagePart> = listOf(
        MessagePart.ToolCall("fix", mapOf<String, JsonElement>("value" to JsonPrimitive("y")), callId = CallIds.mint()),
        MessagePart.ToolCall("finish", mapOf<String, JsonElement>("reason" to JsonPrimitive("done")), callId = CallIds.mint()),
    )

    /** Emits `batches[turn]` per round-trip; past the end, plain text so the loop winds down. */
    private fun executor(captured: MutableList<Prompt>, batches: List<List<MessagePart>>): PromptExecutor =
        PromptExecutor { prompt ->
            flow {
                val turn = captured.size
                captured += prompt
                emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
                (batches.getOrNull(turn) ?: listOf(MessagePart.Text("done"))).forEach { emit(PartReceived(it)) }
                emit(Completed(finishReason = "STOP"))
            }
        }

    private fun Prompt.toolResultTexts(): List<String> =
        content.flatMap { it.parts }
            .filterIsInstance<ToolResultPromptPart>()
            .map { (it.result as? JsonPrimitive)?.content ?: it.result.toString() }

    /** NB: no error-classification argument — the guard must work off the framework default. */
    private fun agent(captured: MutableList<Prompt>, batches: List<List<MessagePart>>) =
        toolCallingAgent<Unit, DeferredFinishTool>(
            llm = fakeLlm, executor = executor(captured, batches), name = "ErrVisAgent", maxTurns = 6,
        ) { _, _, _ -> user { text("test prompt") } }

    @Test
    fun `a finishing batch with an error result is granted one forced round-trip`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = DeferredFinishTool()
        // rt0: error + finish (would finish) → held; rt1: clean fix + finish → finishes.
        executePipeline { execute(agent(captured, listOf(errorBatch(), cleanBatch())), ToolCallingInput(Unit, tool)) }

        assertEquals(2, captured.size, "the errored finishing batch must be granted exactly one extra round-trip")
        assertEquals(DefaultAgentState.FINISHED, tool.currentState, "the invocation finishes on the forced round-trip's clean batch")
        assertEquals(1, tool.errorCalls, "the error tool ran once (rt0)")
        assertEquals(1, tool.fixCalls, "the corrective ran once, on the forced round-trip")

        // The forced round-trip's prompt carries rt0's error result, so the model actually sees it.
        assertTrue(
            captured[1].toolResultTexts().any { it.startsWith("ERROR: boom") },
            "the forced round-trip must re-deliver the hidden error; saw: ${captured[1].toolResultTexts()}",
        )
    }

    @Test
    fun `a clean finishing batch finishes immediately without a forced round-trip`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = DeferredFinishTool()
        executePipeline { execute(agent(captured, listOf(cleanBatch())), ToolCallingInput(Unit, tool)) }

        assertEquals(1, captured.size, "a batch with no matching error must finish on the same round-trip — no hold")
        assertEquals(DefaultAgentState.FINISHED, tool.currentState)
        assertEquals(0, tool.errorCalls)
        assertEquals(1, tool.fixCalls)
    }

    @Test
    fun `a second consecutive errored batch is allowed to finish — the hold is spent once`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = DeferredFinishTool()
        // Both round-trips error-and-finish. rt0 is held; rt1 errors again but is NOT held again —
        // it finishes (with a WARN), so the loop cannot livelock on a persistently-erroring model.
        executePipeline { execute(agent(captured, listOf(errorBatch(), errorBatch())), ToolCallingInput(Unit, tool)) }

        assertEquals(2, captured.size, "exactly one forced round-trip — the second errored batch finishes, not a third hold")
        assertEquals(DefaultAgentState.FINISHED, tool.currentState, "the invocation finishes despite the persistent error")
        assertEquals(2, tool.errorCalls, "both batches ran their error call")
    }

    @Test
    fun `the default matcher's case-insensitive form triggers the hold too`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        // "Error: …" is what the framework itself renders for a tool that threw — the guard must
        // catch it case-insensitively, not only the upper-case "ERROR:" form.
        val tool = DeferredFinishTool(errorText = "Error: kaboom")
        executePipeline { execute(agent(captured, listOf(errorBatch(), cleanBatch())), ToolCallingInput(Unit, tool)) }

        assertEquals(2, captured.size, "the default matcher is case-insensitive on ERROR — the finish must be held")
        assertEquals(DefaultAgentState.FINISHED, tool.currentState)
    }

    // ─── Mid-flow phase transitions: the hold budget is per "place", not per invocation ───────

    enum class FlowState : AgentState { SETUP, WORK, FINISHED }

    /**
     * A deferred-applier [Stateful] over a THREE-phase machine (SETUP → WORK → FINISHED), i.e. one
     * with a MID-FLOW phase change, like the turn planner's SETUP → ACTIONS. Shaped exactly like the
     * pipeline builders: `transitionTo` evaluates the gate immediately and only REGISTERS the intent,
     * which the framework applies at the end of the round-trip in [onAfterToolCalls]; a held
     * round-trip therefore drops the intent (it is cleared in [onRoundTripStart]) and the model must
     * request the transition again.
     *
     * [statesSeen] records the phase each round-trip STARTED in — the direct observable for "was the
     * transition applied or held".
     */
    class DeferredFlowTool : Stateful<FlowState, DeferredFlowTool> {
        override val stateMachine = stateMachine<FlowState, DeferredFlowTool>(initial = FlowState.SETUP) {
            state(FlowState.SETUP, "Setup", "Prepare the plan.") {
                transition(to = FlowState.WORK, name = "start_work", description = "Move on to the work phase.")
            }
            state(FlowState.WORK, "Work", "Do the work.") {
                transition(to = FlowState.FINISHED, name = "finish", description = "Finish the work.")
            }
            terminal(FlowState.FINISHED, "Finished", "Work is complete.")
        }

        private var state = FlowState.SETUP
        override val currentState: FlowState get() = state

        /** Registered-but-not-yet-applied transition of the CURRENT round-trip (deferred applier). */
        private var pending: FlowState? = null

        val statesSeen = mutableListOf<FlowState>()
        var errorCalls = 0
            private set
        var fixCalls = 0
            private set

        @Instrument("Returns an error result (simulates a mutation rejected by a domain gate).")
        fun doError(@InstrumentParam("Any value") value: String): String {
            errorCalls++
            return "ERROR: rejected by the gate"
        }

        @Instrument("A clean corrective call.")
        fun fix(@InstrumentParam("Any value") value: String): String {
            fixCalls++
            return "fixed: $value"
        }

        override suspend fun onRoundTripStart(state: FlowState) {
            pending = null
            statesSeen += state
        }

        override suspend fun onAfterToolCalls(stateAtStart: FlowState) {
            pending?.let { state = it }
        }

        override suspend fun transitionTo(nextState: FlowState): TransitionResult {
            val check = stateMachine.checkTransition(state, nextState, this)
            if (check.success) pending = nextState
            return check
        }
    }

    private fun errorAndTransition(target: FlowState): List<MessagePart> = listOf(
        MessagePart.ToolCall("doError", mapOf<String, JsonElement>("value" to JsonPrimitive("x")), callId = CallIds.mint()),
        MessagePart.ToolCall("transitionTo", mapOf<String, JsonElement>("nextState" to JsonPrimitive(target.name)), callId = CallIds.mint()),
    )

    private fun fixAndTransition(target: FlowState): List<MessagePart> = listOf(
        MessagePart.ToolCall("fix", mapOf<String, JsonElement>("value" to JsonPrimitive("y")), callId = CallIds.mint()),
        MessagePart.ToolCall("transitionTo", mapOf<String, JsonElement>("nextState" to JsonPrimitive(target.name)), callId = CallIds.mint()),
    )

    private fun errorOnly(): List<MessagePart> = listOf(
        MessagePart.ToolCall("doError", mapOf<String, JsonElement>("value" to JsonPrimitive("x")), callId = CallIds.mint()),
    )

    private fun flowAgent(captured: MutableList<Prompt>, batches: List<List<MessagePart>>) =
        toolCallingAgent<Unit, DeferredFlowTool>(
            llm = fakeLlm, executor = executor(captured, batches), name = "FlowAgent", maxTurns = 8,
        ) { _, _, _ -> user { text("test prompt") } }

    /**
     * The baseline for a MID-flow transition: a batch that both errors and requests SETUP → WORK is
     * held, the phase does NOT advance, and the model gets its turn back in SETUP with the rejection
     * visible. (Also held by the pre-fix guard, whose trigger was already "any error in the batch" —
     * kept here as the explicit statement of the mid-flow contract.)
     */
    @Test
    fun `an errored batch requesting a mid-flow transition is held and the phase does not advance`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = DeferredFlowTool()
        executePipeline {
            execute(
                flowAgent(
                    captured,
                    listOf(
                        errorAndTransition(FlowState.WORK),
                        fixAndTransition(FlowState.WORK),
                        fixAndTransition(FlowState.FINISHED),
                    ),
                ),
                ToolCallingInput(Unit, tool),
            )
        }

        assertEquals(
            listOf(FlowState.SETUP, FlowState.SETUP, FlowState.WORK), tool.statesSeen,
            "the errored SETUP→WORK batch must NOT advance the phase; the next round-trip runs in SETUP again",
        )
        assertEquals(FlowState.FINISHED, tool.currentState)
        assertEquals(1, tool.errorCalls)
        assertTrue(
            captured[1].toolResultTexts().any { it.startsWith("ERROR: rejected by the gate") },
            "the forced round-trip must re-deliver the rejection; saw: ${captured[1].toolResultTexts()}",
        )
    }

    /**
     * Anti-livelock, generalised from "one hold per invocation" to "never twice in a row at the same
     * place": the model re-sends the SAME failing batch with the SAME requested transition, so the
     * transition is applied (with a WARN) instead of being held again.
     */
    @Test
    fun `the same transition requested twice in a row with errors is applied instead of held again`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = DeferredFlowTool()
        executePipeline {
            execute(
                flowAgent(
                    captured,
                    listOf(
                        errorAndTransition(FlowState.WORK),
                        errorAndTransition(FlowState.WORK),
                        fixAndTransition(FlowState.FINISHED),
                    ),
                ),
                ToolCallingInput(Unit, tool),
            )
        }

        assertEquals(
            listOf(FlowState.SETUP, FlowState.SETUP, FlowState.WORK), tool.statesSeen,
            "the second consecutive errored batch requesting the same transition must go through — no second hold",
        )
        assertEquals(2, tool.errorCalls, "both batches ran their error call")
        assertEquals(FlowState.FINISHED, tool.currentState)
    }

    /**
     * The live incident (2026-07-30, planner SETUP batch): an earlier errored batch that requested NO
     * transition must NOT eat the hold that the LATER transition-carrying batch needs. Pre-fix the
     * single per-invocation hold was spent on round-trip 0, so round-trip 1 advanced the phase over an
     * unseen rejection.
     */
    @Test
    fun `an earlier errored batch without a transition does not consume the transition hold`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = DeferredFlowTool()
        executePipeline {
            execute(
                flowAgent(
                    captured,
                    listOf(
                        errorOnly(),
                        errorAndTransition(FlowState.WORK),
                        fixAndTransition(FlowState.WORK),
                        fixAndTransition(FlowState.FINISHED),
                    ),
                ),
                ToolCallingInput(Unit, tool),
            )
        }

        assertEquals(
            listOf(FlowState.SETUP, FlowState.SETUP, FlowState.SETUP, FlowState.WORK), tool.statesSeen,
            "the transition-carrying batch gets its own hold — the placeless hold of round-trip 0 must not spend it",
        )
        assertEquals(FlowState.FINISHED, tool.currentState)
        assertEquals(2, tool.errorCalls)
    }

    /**
     * A hold is available at EVERY place, not once per invocation: after an earlier hold at
     * SETUP→WORK has been spent and resolved, an errored batch requesting WORK→FINISHED later in the
     * same invocation is held too. Pre-fix that second batch finished the invocation with the error
     * unseen — the exact blind spot the guard exists for.
     */
    @Test
    fun `a later transition batch gets its own hold`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = DeferredFlowTool()
        executePipeline {
            execute(
                flowAgent(
                    captured,
                    listOf(
                        errorAndTransition(FlowState.WORK),
                        fixAndTransition(FlowState.WORK),
                        errorAndTransition(FlowState.FINISHED),
                        fixAndTransition(FlowState.FINISHED),
                    ),
                ),
                ToolCallingInput(Unit, tool),
            )
        }

        assertEquals(
            listOf(FlowState.SETUP, FlowState.SETUP, FlowState.WORK, FlowState.WORK), tool.statesSeen,
            "the errored WORK→FINISHED batch must be held too — the hold budget is per place, not per invocation",
        )
        assertEquals(FlowState.FINISHED, tool.currentState)
        assertEquals(2, tool.errorCalls)
        assertEquals(2, tool.fixCalls)
    }

    /**
     * A plain tool — NOT a [Stateful]. The framework falls back to `defaultStateful`, which is a
     * DEFERRED applier (its `transitionTo` only registers the pending target; the framework applies
     * it in [Stateful.onAfterToolCalls]). This is the shape of every classifier / one-shot agent in
     * the codebase — and the shape the pre-fix hole lived in: the transition used to apply INSIDE
     * the tool call, so an errored batch that also carried `transitionTo(FINISHED)` ended the
     * invocation at the top of the next loop pass and the forced error-visibility round-trip was
     * silently annulled.
     */
    class PlainErroringTool {
        var errorCalls = 0
            private set
        var fixCalls = 0
            private set

        @Instrument("Returns an error result (simulates a rejected mutation hidden inside a batch).")
        fun doError(@InstrumentParam("Any value") value: String): String {
            errorCalls++
            return "ERROR: boom"
        }

        @Instrument("A clean corrective call.")
        fun fix(@InstrumentParam("Any value") value: String): String {
            fixCalls++
            return "fixed: $value"
        }
    }

    private fun plainAgent(captured: MutableList<Prompt>, batches: List<List<MessagePart>>) =
        toolCallingAgent<Unit, PlainErroringTool>(
            llm = fakeLlm, executor = executor(captured, batches), name = "PlainAgent", maxTurns = 6,
        ) { _, _, _ -> user { text("test prompt") } }

    /**
     * THE invariant this fix exists for: an errored batch carrying a TERMINAL `transitionTo` must
     * not end the invocation — the transition is deferred (held), the agent makes one more real LLM
     * round-trip with the error visible, and only after that clean round does the invocation finish.
     */
    @Test
    fun `an errored batch with a terminal transitionTo is held and finishes only after a clean round`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = PlainErroringTool()
        val erroredFinish = listOf(
            MessagePart.ToolCall("doError", mapOf<String, JsonElement>("value" to JsonPrimitive("x")), callId = CallIds.mint()),
            MessagePart.ToolCall("transitionTo", mapOf<String, JsonElement>("nextState" to JsonPrimitive("FINISHED")), callId = CallIds.mint()),
        )
        val cleanFix = listOf(
            MessagePart.ToolCall("fix", mapOf<String, JsonElement>("value" to JsonPrimitive("y")), callId = CallIds.mint()),
        )

        executePipeline { execute(plainAgent(captured, listOf(erroredFinish, cleanFix)), ToolCallingInput(Unit, tool)) }

        // Exactly 2 round-trips: the hold grants one forced LLM round, and the deferred transition
        // applies at the end of that clean round — not before (pre-fix: 1), not never (maxTurns: 6).
        assertEquals(2, captured.size, "the errored terminal batch must get exactly one forced round-trip before finishing")
        assertEquals(1, tool.errorCalls)
        assertEquals(1, tool.fixCalls, "the corrective ran on the forced round-trip — the invocation had not finished yet")
        assertTrue(
            captured[1].toolResultTexts().any { it.startsWith("ERROR: boom") },
            "the forced round-trip must re-deliver the hidden error; saw: ${captured[1].toolResultTexts()}",
        )
        // The model is told its accepted transition was deferred because of the errors.
        val texts = captured[1].content.flatMap { it.parts }.filterIsInstance<TextPromptPart>().map { it.text }
        assertTrue(
            texts.any { it.contains("DEFERRED") && it.contains("transitionTo") },
            "the forced round-trip's prompt must carry the transition-deferred note; saw: $texts",
        )
    }

    /** Control: a clean batch with a terminal transitionTo still finishes on the same round-trip. */
    @Test
    fun `a clean batch with a terminal transitionTo finishes without a forced round-trip`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = PlainErroringTool()
        val cleanFinish = listOf(
            MessagePart.ToolCall("fix", mapOf<String, JsonElement>("value" to JsonPrimitive("y")), callId = CallIds.mint()),
            MessagePart.ToolCall("transitionTo", mapOf<String, JsonElement>("nextState" to JsonPrimitive("FINISHED")), callId = CallIds.mint()),
        )

        executePipeline { execute(plainAgent(captured, listOf(cleanFinish)), ToolCallingInput(Unit, tool)) }

        assertEquals(1, captured.size, "a clean terminal batch must finish on the same round-trip — no hold")
        assertEquals(0, tool.errorCalls)
        assertEquals(1, tool.fixCalls)
    }
}
