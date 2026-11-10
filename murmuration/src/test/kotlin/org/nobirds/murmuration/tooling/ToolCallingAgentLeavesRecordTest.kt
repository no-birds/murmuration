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
import kotlin.reflect.full.primaryConstructor
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull

/**
 * the record's reader is declared by the
 * instrument's `@`[LeavesRecord] contract — read from the source function the same way
 * `@`[Phases] is (merged across inheritance, see [collectAllAnnotations]) — never measured
 * around the call. [ToolOutcome.declaredReader] is filled at all four of `executeCall`'s
 * returns, and has no default: `null` there is the load-bearing "not declared" class
 * (see [RecordReader]), not "nothing to report".
 */
class ToolCallingAgentLeavesRecordTest {

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    /**
     * [declared] carries its own `@LeavesRecord(WORLD)`. [undeclared] carries none at all —
     * `declaredReader` must come back `null`, not some default work-adjacent value.
     * [phaseOnlyDeclared] is both `@LeavesRecord(WORLD)`-marked and never in-phase, pinning that
     * an EARLY refusal still carries the target instrument's declaration — it is a property of
     * the instrument, not of execution. `transitionTo` is overridden WITHOUT its own annotations
     * (Kotlin does not copy annotations onto overrides) — its outcome must carry the declaration
     * inherited from [Stateful]'s interface method (`StateMachine.kt:87`'s
     * `@LeavesRecord(OWN_PHASE)`), proving the declaration travels the same inheritance path
     * `@Phases`/`@NeverSkip`/`@CantCloseTask` already travel.
     */
    class LeavesRecordTool : Stateful<DefaultAgentState, LeavesRecordTool> {
        override val stateMachine = defaultStateMachine<LeavesRecordTool>()
        override val currentState: DefaultAgentState = DefaultAgentState.WORKING

        @LeavesRecord(RecordReader.WORLD)
        @Instrument("Declares a WORLD reader.")
        fun declared(@InstrumentParam("Any value") value: String): String = "ok"

        @Instrument("No @LeavesRecord at all.")
        fun undeclared(@InstrumentParam("Any value") value: String): String = "ok"

        @Phases("NEVER")
        @LeavesRecord(RecordReader.WORLD)
        @Instrument("Declares WORLD but is never reachable — always PHASE_REFUSED.")
        fun phaseOnlyDeclared(@InstrumentParam("Any value") value: String): String = "unreachable"

        override suspend fun transitionTo(nextState: DefaultAgentState): TransitionResult =
            TransitionResult.Success("advanced")
    }

    private val batch0 = listOf(
        MessagePart.ToolCall("declared", mapOf("value" to JsonPrimitive("d")), callId = CallIds.mint()),
        MessagePart.ToolCall("undeclared", mapOf("value" to JsonPrimitive("u")), callId = CallIds.mint()),
        MessagePart.ToolCall("phaseOnlyDeclared", mapOf("value" to JsonPrimitive("p")), callId = CallIds.mint()),
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

    private class CapturingListener : PipelineEventListener {
        val entries = mutableListOf<PromptExecutionEntry>()

        override suspend fun onEntryReported(pipelineId: UUID, entry: AgentExecutionEntry) {
            if (entry is PromptExecutionEntry) entries += entry
        }
    }

    @Test
    fun `declaredReader comes from the instrument's contract, merged across inheritance, absent when undeclared`() = runBlocking {
        val captured = mutableListOf<Prompt>()
        val tool = LeavesRecordTool()
        val listener = CapturingListener()

        executePipeline(pipelineEventListener = listener) {
            execute(
                toolCallingAgent<Unit, LeavesRecordTool>(
                    llm = fakeLlm, executor = executor(captured), name = "LeavesRecordAgent", maxTurns = 2,
                ) { _, _, _ -> user { text("test prompt") } },
                ToolCallingInput(Unit, tool),
            )
        }

        val rt0 = listener.entries.first { it.toolOutcomes.isNotEmpty() }
        val byName = rt0.toolOutcomes.groupBy { it.name }

        assertEquals(
            RecordReader.WORLD, byName.getValue("declared").single().declaredReader,
            "an instrument's own @LeavesRecord must reach its outcome",
        )
        assertNull(
            byName.getValue("undeclared").single().declaredReader,
            "no @LeavesRecord anywhere — declaredReader must be null, not defaulted",
        )

        val phaseRefused = byName.getValue("phaseOnlyDeclared").single()
        assertEquals(ToolOutcome.Exit.PHASE_REFUSED, phaseRefused.exit)
        assertEquals(
            RecordReader.WORLD, phaseRefused.declaredReader,
            "the declaration is a property of the instrument — an early refusal still carries it",
        )

        val transitionCall = byName.getValue("transitionTo").single()
        assertEquals(
            RecordReader.OWN_PHASE, transitionCall.declaredReader,
            "LeavesRecordTool.transitionTo carries no annotations of its own — the declaration " +
                "must arrive from Stateful's interface method, the same way @NeverSkip/@AllPhases do",
        )
    }

    @Test
    fun `ToolOutcome's declaredReader parameter has no default value`() {
        val parameter = ToolOutcome::class.primaryConstructor!!.parameters.single { it.name == "declaredReader" }
        assertFalse(
            parameter.isOptional,
            "declaredReader must NOT default to null like the neighbouring thoughtSignature — a " +
                "default would let any of executeCall's four returns silently forget to pass it through",
        )
    }
}
