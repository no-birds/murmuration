package org.nobirds.murmuration.tooling

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement
import org.nobirds.murmuration.prompt.CallId
import org.nobirds.murmuration.prompt.ToolOutcomeSerializer
import org.nobirds.murmuration.prompt.ToolResultPromptPart

/**
 * The fact of how a single tool call was resolved by [ToolsExecutor.executeCall] — the return
 * TYPE of that function, not a value written from one of its branches. `executeCall` has four
 * distinct returns (phase-refused, substate-locked, hard-skipped repeat, invoked), and each one
 * constructs a [ToolOutcome] at its own return statement: the compiler refuses to let a fifth
 * branch, added a year from now, exit without one. A fifth means of leaving the function (the
 * "unknown tool" throw) never reaches here at all — the call never happened in any sense the
 * model can act on, so there is nothing to record.
 *
 * Carries no verdict about what the outcome MEANS ("work" / "bookkeeping" / "refusal") — that
 * classification is analysis's job at report-assembly time, over the [isError]/[exit] facts
 * recorded here.
 *
 * `@Serializable`: the snapshot writer embeds this WHOLE object into the record it
 * writes rather than copying its fields one by one, so a field added here later is captured by
 * construction — the same "whole object, not a hand-picked subset" discipline `statistics`
 * already uses inside `PromptExecutionEntry`. `RecordReader`/`Exit` serialize as plain enums
 * without their own annotation (kotlinx.serialization generates their serializers automatically
 * when referenced from a `@Serializable` type); `JsonElement` fields follow the same precedent
 * already used by [org.nobirds.murmuration.prompt.MessagePart.ToolCall.args].
 */
@Serializable(with = ToolOutcomeSerializer::class)
data class ToolOutcome(
    /** Tool name as called by the model — [org.nobirds.murmuration.prompt.ToolCall.name]. */
    val name: String,
    /** FULL call arguments (not the ≤5-key summary a display layer might project). */
    val args: Map<String, JsonElement>,
    /** The rendered result the model actually saw for this call. */
    val rendered: JsonElement,
    /** Whether this call's outcome counts as an error/refusal — see [ErrorMatcher]. */
    val isError: Boolean,
    /** Position of this call within its round-trip's batch (assigned on entry to `executeCall`). */
    val ordinal: Int,
    /** Whether the tool was actually invoked. False for every early-return exit. */
    val invoked: Boolean,
    /** Which of `executeCall`'s four returns produced this outcome. */
    val exit: Exit,
    /**
     * Who reads the record this call left behind — the
     * instrument's [LeavesRecord] contract, `null` when its author left it undeclared. A
     * property of the instrument, not of execution: even an early-return refusal carries its
     * target instrument's declaration.
     *
     * Deliberately WITHOUT a default: unlike [thoughtSignature], `null` here is a load-bearing
     * enum value ("not declared" — see [RecordReader]), not "nothing to report". A default would
     * let every one of `executeCall`'s four returns silently forget to pass it through.
     */
    val declaredReader: RecordReader?,
    val thoughtSignature: String? = null,
    /** Id of the call this outcome answers, copied from [org.nobirds.murmuration.prompt.ToolCall.callId]. */
    val callId: CallId,
) {
    enum class Exit {
        /** The call target a tool not available in the current phase — never invoked. */
        PHASE_REFUSED,

        /** The call target a tool locked by the current dynamic substate — never invoked. */
        LOCKED,

        /** A third+ identical repeat — the stored (non-error) result was replayed, not invoked. */
        REPEAT_SKIPPED,

        /** The tool actually ran; [isError] carries its real success/failure verdict. */
        INVOKED,
    }
}

/**
 * Projects a [ToolOutcome] into the prompt part the conversation history actually carries. This
 * is the ONLY place that builds a [ToolResultPromptPart] from an outcome — batch assembly
 * (`ToolCallingAgent.executeTurn`), not any of `executeCall`'s four return points.
 */
fun ToolOutcome.toPromptPart(): ToolResultPromptPart = ToolResultPromptPart(name, rendered, callId, thoughtSignature)
