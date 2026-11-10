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
import org.nobirds.murmuration.prompt.ToolResultPromptPart
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * A round-trip that mechanically repeats the previous one's call (same tool, same args, same
 * result) gets [REPEATED_CALL_ADVISORY] appended to its tool result, so the model sees in-band
 * that the effect is already applied and it should stop re-issuing the working call. Regression
 * for the planner loop where `setCurrentPlayers(['silas-gray'])` was re-issued three round-trips
 * running instead of being closed with `completeTask`.
 */
class ToolCallingAgentRepeatedCallTest {

    class EchoTool {
        /** Counts real invocations so a test can assert a hard-skipped repeat never ran the tool. */
        var invocations = 0
        @Instrument("Echo a value back — returns a constant OK regardless of the argument.")
        fun echo(@InstrumentParam("Any value") value: String): String {
            invocations++
            return "OK"
        }
    }

    /** Always returns an error-prefixed result — exercises the "never skip a stored error" guard. */
    class FailingTool {
        var invocations = 0
        @Instrument("Always fails with an error-prefixed message.")
        fun act(@InstrumentParam("Any value") value: String): String {
            invocations++
            return "ERROR: still blocked"
        }
    }

    /**
     * A @NeverSkip control gate whose result changes with external state: the first [blockedFor]
     * calls report a refusal (a [TransitionResult.Failure] whose wording carries NO error prefix —
     * the exact case the honest classification must catch), then it "succeeds". Models the
     * `transitionTo` gate that refuses while a task is open and passes once it is closed.
     */
    class GateTool(private val blockedFor: Int) {
        var invocations = 0
        @Instrument("Transition gate — refuses until the blocker clears, then succeeds.")
        @NeverSkip
        fun gate(@InstrumentParam("Target") target: String): TransitionResult {
            invocations++
            return if (invocations <= blockedFor)
                TransitionResult.Failure("Close the required task, then retry the transition to $target")
            else TransitionResult.Success("OK — transitioned to $target")
        }
    }

    /**
     * A @NeverSkip control gate that ALWAYS succeeds with a byte-identical result — models the
     * legitimate case where the SAME transition fires more than once in one run (e.g. the nano
     * agent's WORKING→REVIEW happening twice: REVIEW finds a finding → WORKING → fix → REVIEW
     * again), both returning the default `"Successful transition to REVIEW"`. The identical
     * SUCCESSFUL repeat must NOT be clobbered with the false "REJECTED, nothing applied" advisory,
     * and must never be hard-skipped. Regression for the GATE-branch bug (mirror of a0214673 for
     * MUTATION): the old code appended [REPEATED_CALL_REFUSAL_ADVISORY] to any GATE repeat,
     * ignoring success.
     */
    class SuccessGateTool {
        var invocations = 0
        @Instrument("Transition gate — always succeeds with the same message.")
        @NeverSkip
        fun gate(@InstrumentParam("Target") target: String): TransitionResult {
            invocations++
            return TransitionResult.Success("Successful transition to $target")
        }
    }

    /**
     * A @NeverSkip positional mutator whose result is CONSTANT across identical calls (models
     * `removePart(index)`: `removeAt` shifts the list, so the same `(index)` legitimately deletes
     * a DIFFERENT element each time, yet the returned value never changes). Without @NeverSkip the
     * third identical issue would be hard-skipped as a no-op, dropping a legitimate deletion —
     * the repeat is neither a no-op nor a rejection.
     */
    class ConstantNeverSkipMutator {
        var invocations = 0
        @Instrument("Positional mutator — deletes at an index; returns a constant result.")
        @NeverSkip
        fun removeAt(@InstrumentParam("Index") index: String): String {
            invocations++
            return "removed"
        }
    }

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    /**
     * Emits one `echo` tool call per round-trip with the argument taken from [argsPerTurn]
     * (cycling), capturing the request [Prompt] of every round-trip into [captured] so the test
     * can inspect what tool results the model was shown.
     */
    private fun echoExecutor(argsPerTurn: List<String>, captured: MutableList<Prompt>) = PromptExecutor { prompt ->
        flow {
            val turn = captured.size
            captured += prompt
            emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
            val value = argsPerTurn[turn.coerceAtMost(argsPerTurn.lastIndex)]
            emit(PartReceived(MessagePart.ToolCall("echo", mapOf<String, JsonElement>("value" to JsonPrimitive(value)), callId = CallIds.mint())))
            emit(Completed(finishReason = "STOP"))
        }
    }

    private fun Prompt.toolResultTexts(): List<String> =
        content.flatMap { it.parts }
            .filterIsInstance<ToolResultPromptPart>()
            .map { (it.result as? JsonPrimitive)?.content ?: it.result.toString() }

    @Test
    fun `identical call across round-trips gets a repeat advisory on its result`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        // Three round-trips all calling echo("x"): rt1 legit, rt2 repeats rt1, rt3 repeats rt2.
        val agent = toolCallingAgent<Unit, EchoTool>(
            llm = fakeLlm, executor = echoExecutor(listOf("x", "x", "x"), captured), name = "EchoAgent",
            maxTurns = 3,
        ) { _, _, _ -> user { text("test prompt") } }
        executePipeline { execute(agent, ToolCallingInput(Unit, EchoTool())) }

        assertEquals(3, captured.size, "three round-trips ran")

        // rt2's prompt shows rt1's result with NO advisory (rt1 was the first, legit call).
        assertTrue(
            captured[1].toolResultTexts().none { it.contains(REPEATED_CALL_ADVISORY) },
            "the first call must not be flagged as a repeat",
        )
        // rt3's prompt shows rt2's result WITH the advisory (rt2 repeated rt1 identically).
        val rt3Results = captured[2].toolResultTexts()
        assertTrue(
            rt3Results.any { it.contains(REPEATED_CALL_ADVISORY) },
            "the repeated call's result must carry the advisory; saw: $rt3Results",
        )
        // The original result is preserved alongside the advisory.
        assertTrue(rt3Results.any { it.startsWith("OK") }, "original result text is kept")
    }

    /**
     * The first identical repeat gets the soft advisory (and still runs); from the SECOND
     * identical repeat on the call is HARD-SKIPPED — the tool is not invoked and the stored
     * result is returned prefixed with [REPEATED_CALL_NOOP]. Regression for the live turn where
     * the model ignored 12 advisories and kept re-issuing `updateToken`/`updateAction`.
     */
    @Test
    fun `third identical call is hard-skipped as a no-op and the tool is not re-invoked`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = EchoTool()
        // Four round-trips all calling echo("x"): rt1 runs, rt2 runs+advisory, rt3 & rt4 skipped.
        val agent = toolCallingAgent<Unit, EchoTool>(
            llm = fakeLlm, executor = echoExecutor(listOf("x", "x", "x", "x"), captured), name = "EchoAgent",
            maxTurns = 4,
        ) { _, _, _ -> user { text("test prompt") } }
        executePipeline { execute(agent, ToolCallingInput(Unit, tool)) }

        assertEquals(4, captured.size, "four round-trips ran")
        // The tool actually executed only twice (rt1, rt2); rt3 and rt4 were short-circuited.
        assertEquals(2, tool.invocations, "the tool must not be invoked for the hard-skipped repeats")
        // rt3's prompt shows rt2's result WITH the soft advisory (rt2 = first repeat, still ran).
        assertTrue(
            captured[2].toolResultTexts().any { it.contains(REPEATED_CALL_ADVISORY) },
            "the first repeat carries the advisory; saw: ${captured[2].toolResultTexts()}",
        )
        // rt4's prompt shows rt3's result as a NO-OP (rt3 = second repeat, hard-skipped).
        val rt4Results = captured[3].toolResultTexts()
        assertTrue(
            rt4Results.any { it.contains(REPEATED_CALL_NOOP) },
            "the hard-skipped repeat carries the NO-OP note; saw: $rt4Results",
        )
        // The original result is preserved alongside the NO-OP note.
        assertTrue(rt4Results.any { it.startsWith("OK") }, "original result text is kept on the no-op")
    }

    /** Emits one call to [toolName] with a constant arg every round-trip, capturing each request. */
    private fun constantCallExecutor(toolName: String, argName: String, argValue: String, captured: MutableList<Prompt>) =
        PromptExecutor { prompt ->
            flow {
                captured += prompt
                emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
                emit(PartReceived(MessagePart.ToolCall(toolName, mapOf<String, JsonElement>(argName to JsonPrimitive(argValue)), callId = CallIds.mint())))
                emit(Completed(finishReason = "STOP"))
            }
        }

    /**
     * A stored ERROR/refusal result is NEVER hard-skipped: an error is not an applied effect, and
     * the condition it reports may clear. Every round-trip re-runs the tool, and an identical
     * repeat carries the honest [REPEATED_CALL_REFUSAL_ADVISORY], never the "effect applied"
     * [REPEATED_CALL_ADVISORY] nor the [REPEATED_CALL_NOOP].
     */
    @Test
    fun `an error result is never hard-skipped and carries the refusal advisory`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = FailingTool()
        val agent = toolCallingAgent<Unit, FailingTool>(
            llm = fakeLlm, executor = constantCallExecutor("act", "value", "x", captured), name = "FailAgent",
            maxTurns = 4,
        ) { _, _, _ -> user { text("test prompt") } }
        executePipeline { execute(agent, ToolCallingInput(Unit, tool)) }

        assertEquals(4, captured.size, "four round-trips ran")
        // The error result is never skipped: the tool ran on every round-trip.
        assertEquals(4, tool.invocations, "an error result must be re-run, never hard-skipped")
        // No round-trip ever showed a NO-OP note (that would mean an error got hard-skipped).
        assertFalse(
            captured.any { p -> p.toolResultTexts().any { it.contains(REPEATED_CALL_NOOP) } },
            "an error must never be presented as a hard-skipped no-op",
        )
        // The first repeat (rt2, shown in rt3's request) carries the honest refusal advisory,
        // not the misleading "effect already applied" one.
        val rt3 = captured[2].toolResultTexts()
        assertTrue(rt3.any { it.contains(REPEATED_CALL_REFUSAL_ADVISORY) }, "refusal advisory expected; saw: $rt3")
        assertTrue(rt3.none { it.contains(REPEATED_CALL_ADVISORY) }, "the 'effect applied' advisory must not be used for an error")
    }

    /**
     * A @NeverSkip control gate (e.g. `transitionTo`) is re-run on every repeat and re-evaluates
     * its gate against current state — so once its blocker clears it succeeds, instead of a stale
     * refusal being frozen and replayed forever. This is the exact planner bug: `transitionTo`
     * refused while a checklist task was open, then hard-skipped even after the task was closed.
     */
    @Test
    fun `a NeverSkip gate is re-run after its blocker clears and is never hard-skipped`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        // Blocked for the first two calls, then succeeds — mirrors a task being closed at rt3.
        val tool = GateTool(blockedFor = 2)
        val agent = toolCallingAgent<Unit, GateTool>(
            llm = fakeLlm, executor = constantCallExecutor("gate", "target", "WORKING", captured), name = "GateAgent",
            maxTurns = 4,
        ) { _, _, _ -> user { text("test prompt") } }
        executePipeline { execute(agent, ToolCallingInput(Unit, tool)) }

        assertEquals(4, captured.size, "four round-trips ran")
        // Never hard-skipped: the gate ran on every round-trip (a plain idempotent tool would have
        // been skipped from the 3rd identical call).
        assertEquals(4, tool.invocations, "a @NeverSkip gate must run on every repeat")
        assertFalse(
            captured.any { p -> p.toolResultTexts().any { it.contains(REPEATED_CALL_NOOP) } },
            "a @NeverSkip gate must never be hard-skipped",
        )
        // rt3 actually ran and produced the fresh success — visible in rt4's request. A frozen
        // stale refusal (the old bug) would show the "Close the required task" text here instead.
        val rt4 = captured[3].toolResultTexts()
        assertTrue(rt4.any { it.contains("OK — transitioned to WORKING") }, "rt3 must re-evaluate and succeed; saw: $rt4")
        // The identical refusal repeat (rt2, shown in rt3's request) uses the honest refusal advisory.
        val rt3 = captured[2].toolResultTexts()
        assertTrue(rt3.any { it.contains(REPEATED_CALL_REFUSAL_ADVISORY) }, "refusal advisory expected on repeat; saw: $rt3")
    }

    /**
     * A @NeverSkip transition gate that legitimately fires the SAME transition twice, returning a
     * byte-identical SUCCESS both times, must carry NO repeat note (neither the false "REJECTED,
     * nothing applied" [REPEATED_CALL_REFUSAL_ADVISORY] nor the "effect already applied"
     * [REPEATED_CALL_ADVISORY]) and must never be hard-skipped. This is the gate-branch bug: the
     * old code appended the refusal advisory to EVERY gate repeat regardless of success, so a
     * legitimate second WORKING→REVIEW was told it had been rejected. The fix makes
     * [Stateful.transitionTo] return a [TransitionResult] so a `Success` repeat is classified as
     * NOT an error.
     */
    @Test
    fun `a NeverSkip GATE with identical SUCCESS results carries no note and is never hard-skipped`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = SuccessGateTool()
        val agent = toolCallingAgent<Unit, SuccessGateTool>(
            llm = fakeLlm, executor = constantCallExecutor("gate", "target", "REVIEW", captured), name = "SuccessGateAgent",
            maxTurns = 4,
        ) { _, _, _ -> user { text("test prompt") } }
        executePipeline { execute(agent, ToolCallingInput(Unit, tool)) }

        assertEquals(4, captured.size, "four round-trips ran")
        // A @NeverSkip gate runs on every repeat — never hard-skipped, even on identical success.
        assertEquals(4, tool.invocations, "a @NeverSkip gate must run on every identical repeat")
        assertFalse(
            captured.any { p -> p.toolResultTexts().any { it.contains(REPEATED_CALL_NOOP) } },
            "a @NeverSkip gate must never be hard-skipped",
        )
        // The identical, SUCCESSFUL repeat (rt2, shown in rt3's request) carries NO note — neither
        // the false "rejected" advisory nor the false "changes nothing" one.
        val rt3 = captured[2].toolResultTexts()
        assertTrue(rt3.none { it.contains(REPEATED_CALL_REFUSAL_ADVISORY) }, "a successful GATE repeat must NOT be told it was rejected; saw: $rt3")
        assertTrue(rt3.none { it.contains(REPEATED_CALL_ADVISORY) }, "a successful GATE repeat must NOT get the 'effect applied' advisory either; saw: $rt3")
        // The original success text is preserved untouched.
        assertTrue(rt3.any { it.startsWith("Successful transition to REVIEW") }, "original success text is kept, with no note appended")
    }

    /**
     * A @NeverSkip(MUTATION) mutator with a CONSTANT result must never be hard-skipped, even
     * though its (name,args,result) is identical every round-trip. This is the `removePart(index)`
     * case: the result never changes but the effect (delete the element now at `index`) is real
     * each time — hard-skipping the 3rd identical issue would silently drop a legitimate deletion.
     * The existing GateTool test only covers a @NeverSkip(GATE) tool whose result CHANGES; this
     * pins the constant-result MUTATION variant that the annotation was added to protect.
     *
     * Regression for a0214673: a legitimate repeated `removePart(index=2)` came back "OK" (the
     * effect WAS applied) but got the [REPEATED_CALL_REFUSAL_ADVISORY] note claiming it was
     * "REJECTED... nothing was applied" — a lie. A MUTATION repeat isn't a no-op ("changes
     * nothing" is false — a different element mutated) or a rejection ("still blocked" is false —
     * it succeeded), so a successful identical repeat gets NO note at all.
     */
    @Test
    fun `a NeverSkip MUTATION with a constant result is never hard-skipped and carries no repeat note`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = ConstantNeverSkipMutator()
        val agent = toolCallingAgent<Unit, ConstantNeverSkipMutator>(
            llm = fakeLlm, executor = constantCallExecutor("removeAt", "index", "4", captured), name = "RemoveAgent",
            maxTurns = 4,
        ) { _, _, _ -> user { text("test prompt") } }
        executePipeline { execute(agent, ToolCallingInput(Unit, tool)) }

        assertEquals(4, captured.size, "four round-trips ran")
        // Ran on every identical repeat — a plain idempotent tool would be skipped from the 3rd.
        assertEquals(4, tool.invocations, "a positional @NeverSkip mutator must run on every identical repeat")
        // Never surfaced as a hard-skipped no-op.
        assertFalse(
            captured.any { p -> p.toolResultTexts().any { it.contains(REPEATED_CALL_NOOP) } },
            "a @NeverSkip mutator must never be hard-skipped",
        )
        // The identical, SUCCESSFUL repeat carries NO note at all — neither the false "rejected"
        // advisory nor the false "changes nothing" one.
        val rt3 = captured[2].toolResultTexts()
        assertTrue(rt3.none { it.contains(REPEATED_CALL_REFUSAL_ADVISORY) }, "a successful MUTATION repeat must NOT be told it was rejected; saw: $rt3")
        assertTrue(rt3.none { it.contains(REPEATED_CALL_ADVISORY) }, "the 'effect applied, changes nothing' advisory must not be used for a MUTATION repeat either; saw: $rt3")
        // The original result is preserved untouched.
        assertTrue(rt3.any { it.startsWith("removed") }, "original result text is kept, with no note appended")
    }

    @Test
    fun `distinct arguments each round-trip are not flagged`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        // Different argument every round-trip → different call → never a repeat.
        val agent = toolCallingAgent<Unit, EchoTool>(
            llm = fakeLlm, executor = echoExecutor(listOf("a", "b", "c"), captured), name = "EchoAgent",
            maxTurns = 3,
        ) { _, _, _ -> user { text("test prompt") } }
        executePipeline { execute(agent, ToolCallingInput(Unit, EchoTool())) }

        assertEquals(3, captured.size, "three round-trips ran")
        assertFalse(
            captured.any { p -> p.toolResultTexts().any { it.contains(REPEATED_CALL_ADVISORY) } },
            "distinct calls must never be flagged as repeats",
        )
    }

    /**
     * The window spans the whole invocation, not just the immediately-previous round-trip: an
     * alternating stall `a, b, a, b` — where each call differs from its *neighbour* but repeats
     * a call two round-trips back — is flagged from its first recurrence on. This is the exact
     * pattern that a prev-round-trip-only comparison missed (the MapBattle loop ping-ponging
     * `removeFromBattle`/`updateToken`).
     */
    @Test
    fun `alternating repeats are flagged across the whole invocation`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        // echo returns a constant "OK", so a==a two round-trips apart is an identical signature.
        val agent = toolCallingAgent<Unit, EchoTool>(
            llm = fakeLlm, executor = echoExecutor(listOf("a", "b", "a", "b"), captured), name = "EchoAgent",
            maxTurns = 4,
        ) { _, _, _ -> user { text("test prompt") } }
        executePipeline { execute(agent, ToolCallingInput(Unit, EchoTool())) }

        assertEquals(4, captured.size, "four round-trips ran")
        // rt2 (shows rt1's `a`) and rt3 (shows rt2's `b`) — both first-time calls, no advisory.
        assertTrue(
            captured[1].toolResultTexts().none { it.contains(REPEATED_CALL_ADVISORY) },
            "the first `a` must not be flagged",
        )
        assertTrue(
            captured[2].toolResultTexts().none { it.contains(REPEATED_CALL_ADVISORY) },
            "the first `b` must not be flagged",
        )
        // rt4 shows rt3's `a`, which repeats rt1's `a` two round-trips back → advisory. A
        // prev-round-trip-only window (rt3 vs rt2=`b`) would have missed this.
        assertTrue(
            captured[3].toolResultTexts().any { it.contains(REPEATED_CALL_ADVISORY) },
            "the second `a` (repeat of rt1) must carry the advisory; saw: ${captured[3].toolResultTexts()}",
        )
    }

    /**
     * A text-only round-trip (no tool calls) must not wipe the accumulated signatures: a repeat
     * that straddles such a round-trip is still caught: a text-only round-trip never touches
     * [RepeatedCallTracker], so the accumulated (name,args)→result memory survives it.
     */
    @Test
    fun `text-only round-trip does not reset the repeat window`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        // rt1: echo("x"); rt2: text only; rt3: echo("x") again (the straddling repeat); rt4:
        // echo("x") once more so rt3's flagged result is committed and visible in rt4's request.
        val executor = PromptExecutor { prompt ->
            flow {
                val turn = captured.size
                captured += prompt
                emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
                if (turn == 1) {
                    emit(PartReceived(MessagePart.Text("just thinking, no tools")))
                } else {
                    emit(PartReceived(MessagePart.ToolCall("echo", mapOf<String, JsonElement>("value" to JsonPrimitive("x")), callId = CallIds.mint())))
                }
                emit(Completed(finishReason = "STOP"))
            }
        }
        val agent = toolCallingAgent<Unit, EchoTool>(
            llm = fakeLlm, executor = executor, name = "EchoAgent", maxTurns = 4,
        ) { _, _, _ -> user { text("test prompt") } }
        executePipeline { execute(agent, ToolCallingInput(Unit, EchoTool())) }

        assertEquals(4, captured.size, "four round-trips ran")
        // rt4's request shows rt3's result; rt3's echo("x") repeats rt1's despite the text-only
        // rt2 in between → the accumulator survived the empty round-trip and rt3 is flagged.
        assertTrue(
            captured[3].toolResultTexts().any { it.contains(REPEATED_CALL_ADVISORY) },
            "repeat straddling a text-only round-trip must still be flagged; saw: ${captured[3].toolResultTexts()}",
        )
    }
}
