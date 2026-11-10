package org.nobirds.murmuration.tooling

import java.util.UUID

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive
import org.nobirds.murmuration.retry.RetryPolicy
import org.nobirds.murmuration.retry.RetrySession
import org.nobirds.murmuration.retry.abortsAgent
import org.nobirds.murmuration.retry.rethrowIfAborts
import org.nobirds.murmuration.agent.generateAgentName
import org.nobirds.murmuration.agent.Agent
import org.nobirds.murmuration.agent.AgentEventListener
import org.nobirds.murmuration.agent.CallLimitDecision
import org.nobirds.murmuration.agent.ErrorMatcher
import org.nobirds.murmuration.agent.ExecutionContext
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.llm.cacheKind
import org.nobirds.murmuration.prompt.ExecutionTextPart
import org.nobirds.murmuration.prompt.InstrumentDeclaration
import org.nobirds.murmuration.prompt.InstrumentsDeclaration
import org.nobirds.murmuration.prompt.MalformedResponseException
import org.nobirds.murmuration.prompt.ParsePhase
import org.nobirds.murmuration.prompt.PartKind
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionException
import org.nobirds.murmuration.prompt.errorCode
import org.nobirds.murmuration.prompt.finishReasonOrNull
import org.nobirds.murmuration.agent.PromptExecutionEntry
import org.nobirds.murmuration.prompt.PromptCacheStrategy
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentBuilder
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptMessage
import org.nobirds.murmuration.prompt.PromptExecutionStatistics
import org.nobirds.murmuration.prompt.buildStatistics
import org.nobirds.murmuration.prompt.StreamDiagnostics
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.agent.PromptExecutorException
import org.nobirds.murmuration.prompt.PromptGenerator
import org.nobirds.murmuration.prompt.PromptPart
import org.nobirds.murmuration.prompt.plus
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.ToolCall
import org.nobirds.murmuration.prompt.ToolCallPromptPart
import org.nobirds.murmuration.prompt.toToolCall
import org.nobirds.murmuration.prompt.ToolResultPromptPart
import org.nobirds.murmuration.prompt.ToolsPromptSettings
import org.nobirds.murmuration.internal.logger
import org.nobirds.murmuration.prompt.mergeWith
import org.nobirds.murmuration.agent.AgentExecutionIdentity
import org.nobirds.murmuration.prompt.system
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.toJsonElement
import org.nobirds.murmuration.prompt.update
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.agent.withIteration
import org.nobirds.murmuration.agent.withState
import org.nobirds.murmuration.prompt.InputTokenStatistics
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.OutputTokenStatistics
import org.nobirds.murmuration.prompt.PromptExecutionEvent.CacheCreated
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Failed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutionEvent.StreamResponded
import org.nobirds.murmuration.prompt.PromptExecutionEvent.UsageDelta
import org.nobirds.murmuration.prompt.PromptExecutionTokenStatistics
import org.nobirds.murmuration.prompt.plus
import org.nobirds.murmuration.agent.ExecutionNature
import org.nobirds.murmuration.agent.reportCacheCreate
import java.lang.reflect.InvocationTargetException
import java.util.concurrent.atomic.AtomicInteger
import kotlin.time.TimeSource

data class ToolCallingInput<In, Tool: Any>(val input: In, val tool: Tool)

/**
 * Section id for the phase description + per-phase tools allowlist. Both blocks change together
 * on a state change — per-content pruning replaces the whole user message on the new phase.
 */
internal const val PHASE_DESCRIPTION_SECTION = "phase-description"

/**
 * Section id for the accumulated model+tool exchanges. A fresh snapshot already reflects
 * the total effect, so older exchanges become token ballast and are dropped by
 * per-content pruning when a new exchange appears on the next round-trip.
 */
internal const val TOOL_EXCHANGE_HISTORY_SECTION = "tool-exchange-history"

/**
 * Snapshot of one tool-calling loop iteration: the accumulated text parts, the announced tool calls
 * and the tool results already executed in parallel, plus the final statistics. Constructed
 * after the `Completed` event and `awaitAll()` of the parallel tool jobs.
 */
private data class TurnResult(
    val parts: List<MessagePart>,
    val toolResults: List<ToolResultPromptPart>,
    val toolOutcomes: List<ToolOutcome>,
    val stats: PromptExecutionStatistics,
    val toolsExecutor: ToolsExecutor,
    val llm: Llm,
) {
    /** Text and thought blocks of the turn in arrival order, a projection of [parts]. */
    val textParts: List<ExecutionTextPart>
        get() = parts.mapNotNull {
            when (it) {
                is MessagePart.Text -> ExecutionTextPart(it.text, isThought = false)
                is MessagePart.Thought -> ExecutionTextPart(it.text, it.signature, isThought = true)
                else -> null
            }
        }

    /** Tool calls of the turn in arrival order, a projection of [parts]. */
    val toolCalls: List<ToolCall>
        get() = parts.filterIsInstance<MessagePart.ToolCall>().map { it.toToolCall() }

    fun asContent() = toolsExecutor.buildContents(
        parts = parts,
        toolResults = toolResults,
        historyCacheStrategy = CachePlacement.of(llm.cacheKind).historyCacheStrategy,
    )
}

/**
 * Cross-round-trip memory of every executed tool call within ONE agent invocation, keyed by
 * (tool name, arguments). It drives two escalating responses to a model that mechanically
 * re-issues a call whose effect is already applied (the classic tool-calling stall where the
 * model believes a follow-up must "wait" for a call it has already made, or a builder
 * ping-ponging the same setters for a task it has no tool to fulfil):
 *
 *  - the FIRST identical repeat (same name+args returning the same result as before) gets
 *    [REPEATED_CALL_ADVISORY] appended to its result — a soft in-band hint, the call still runs;
 *  - from the SECOND identical repeat on the call is HARD-SKIPPED: the tool is NOT invoked and
 *    the stored result is returned prefixed with [REPEATED_CALL_NOOP]. Advisory alone proved
 *    insufficient on live runs (the model ignored 12 advisories in one turn), so the third+
 *    occurrence is cut mechanically.
 *
 * Why the skip is keyed on (name,args) and gated on ≥2 prior *identical* results: the skip
 * decision is made BEFORE execution, when this call has no result yet — so the key cannot
 * include a result. Two prior executions that returned the *same* result are the safety gate:
 * a call proven to return an identical value twice is idempotent in fact, so short-circuiting
 * the third is safe. A non-deterministic call (a fresh `roll`) returns a *different* value on
 * its repeat, which [record] treats as a reset (new result, count 1), so it never reaches the
 * skip threshold and is never suppressed. Residual false positive: a random roll that happens
 * to return the same value twice in a row would have a third identical-args issue skipped —
 * rare, and re-issuing an identical roll three times is itself a degenerate pattern the model
 * should not be doing.
 *
 * Thread-safe (methods synchronized) so a [ParallelToolCallScheduler] executing a round-trip's
 * calls concurrently cannot corrupt the counts; the default sequential scheduler serialises
 * calls anyway.
 */
internal class RepeatedCallTracker {
    private data class Key(val name: String, val args: Map<String, JsonElement>)
    private class Entry(var result: JsonElement, var identicalCount: Int, var isError: Boolean)

    private val entries = HashMap<Key, Entry>()

    /**
     * The stored result to short-circuit this call with (returning it WITHOUT invoking the
     * tool), or null to execute normally. Non-null once the same (name,args) has already been
     * executed at least twice with an identical, NON-ERROR result — its effect is proven applied
     * and idempotent, so the third+ issue is a no-op.
     *
     * A stored result that the caller flagged as an error/refusal ([record] `isError`) is NEVER
     * skipped: an error is not an "applied effect", and the condition it reports may have been
     * cleared by other calls since it was recorded (e.g. a blocking task closed). Re-running lets
     * the tool re-evaluate against current state instead of freezing a stale refusal forever —
     * the live bug where a `transitionTo` refused while a checklist was open kept being skipped
     * (replaying the stale refusal) even after the checklist had been closed.
     */
    @Synchronized
    fun skipResultFor(name: String, args: Map<String, JsonElement>): JsonElement? {
        val entry = entries[Key(name, args)] ?: return null
        if (entry.isError) return null
        return if (entry.identicalCount >= 2) entry.result else null
    }

    /**
     * Record an executed call's ORIGINAL (un-annotated) result and return true if it repeats a
     * previous identical (name,args,result) — the caller then appends the appropriate advisory.
     * A result that differs from the stored one resets the entry (a non-deterministic call), so
     * it is neither flagged nor ever skipped. [isError] tags whether this result is an
     * error/refusal (classified by the caller's `ErrorMatcher`); a stored error is never
     * hard-skipped (see [skipResultFor]).
     */
    @Synchronized
    fun record(name: String, args: Map<String, JsonElement>, result: JsonElement, isError: Boolean): Boolean {
        val key = Key(name, args)
        val entry = entries[key]
        return when {
            entry == null -> { entries[key] = Entry(result, 1, isError); false }
            entry.result == result -> { entry.identicalCount++; entry.isError = isError; true }
            else -> { entry.result = result; entry.identicalCount = 1; entry.isError = isError; false }
        }
    }
}

internal const val REPEATED_CALL_ADVISORY: String =
    "⚠️ Repeated call: you already issued this exact tool call with the same arguments " +
        "earlier in this turn, and it returned the same result — its effect is already applied " +
        "and repeating it changes nothing. A working call does NOT need to be re-issued before you " +
        "close a task or move on: the previous result is already available above. If this call was " +
        "meant to resolve a task, call the appropriate completion tool now; otherwise continue with " +
        "the next step instead of repeating this call."

internal const val REPEATED_CALL_NOOP: String =
    "⚠️ NO-OP: this exact tool call already ran earlier in this turn with the same result — its " +
        "effect is applied and this repeat was IGNORED (the tool was not invoked again). Stop " +
        "re-issuing it: if you are stuck, escalate or close the task with the appropriate tool " +
        "instead of repeating this call."

/**
 * Appended when an identical repeat's stored result is an error/refusal. For a control-flow gate
 * this is driven by an HONEST classification: [Stateful.transitionTo] returns a [TransitionResult]
 * whose [TransitionResult.Failure] is recognised as a refusal directly (via
 * [TransitionResult.success]), without hoping its wording matches the [ErrorMatcher] prefix
 * convention — so "same refusal twice" means "still blocked" and the model is told to fix the
 * blocker (named in the result above) instead of re-issuing the rejected call. Unlike
 * [REPEATED_CALL_ADVISORY] it does NOT claim the effect is applied — because it isn't: the call
 * was REJECTED. Such a call is never hard-skipped: the blocker may clear (a task closed, a phase
 * gate satisfied), and the next issue must actually run to re-evaluate against current state.
 *
 * NOT appended to a [NeverSkip] repeat whose result is NOT an error: a legitimate re-application
 * (a positional mutator on a shifted target, or a byte-identical SUCCESSFUL transition fired
 * twice in one run) is neither a rejection nor a no-op, so it carries no note.
 */
internal const val REPEATED_CALL_REFUSAL_ADVISORY: String =
    "⚠️ Repeated rejection: you already issued this exact tool call earlier in this turn and it " +
        "was REJECTED with the same result — nothing was applied, and repeating it verbatim changes " +
        "nothing. The blocking condition named in the result above is NOT yet resolved: address it " +
        "first (e.g. close the required task with the completion tool, or satisfy the phase rule), " +
        "THEN retry. Do NOT re-create work under new ids and do NOT just re-send the same call."

/**
 * [forced error-visibility] Stand-in "place" for an errored batch that requested no transition the
 * framework can see (see [ToolCallingAgent.errorMatcher]): such holds cannot be told apart from one
 * another, so they share one identity and one per-invocation budget.
 */
private const val PLACELESS_HOLD: String = "<no-requested-transition>"

/**
 * [forced error-visibility] Stable identity of a transition-requesting call: tool name plus its
 * arguments in a deterministic (key-sorted) order, so the SAME requested transition produces the
 * same string on a later round-trip regardless of the argument order the provider happened to emit.
 */
private fun ToolCall.transitionSignature(): String = "$name(${args.toSortedMap()})"

/** Render a tool result [JsonElement] as its plain text form (a string primitive unquoted). */
private fun JsonElement.renderText(): String =
    (this as? JsonPrimitive)?.let { if (it.isString) it.content else it.toString() } ?: this.toString()

/** Render a tool result as text and append an out-of-band note ([REPEATED_CALL_ADVISORY] / [REPEATED_CALL_NOOP]). */
private fun JsonElement.appendNote(note: String): JsonElement = JsonPrimitive("${renderText()}\n\n$note")

class ToolCallingAgent<In, Tool: Any>(
    private val promptExecutor: PromptExecutor,
    private val promptGenerator: PromptGenerator<ToolCallingInput<In, Tool>>,
    name: String = "ToolCallingAgent",
    private val maxTurns: Int = 10,
    private val maxTurnsFn: ((In) -> Int)? = null,
    private val eventListener: AgentEventListener? = null,
    /**
     * Caching strategy for the **framework instructions** (CRITICAL tool-calling
     * instruction + STATE MACHINE FLOW). The content is constant for the lifetime of the agent instance,
     * so the default is `PIPELINE` — the instructions enter the cache prefix once and are
     * reused across all round-trips. Pass `NO_CACHE` if you use the agent outside a pipeline
     * context or know that caching is not justified (for example, a one-off
     * call).
     */
    private val frameworkInstructionsCacheStrategy: PromptCacheStrategy = PromptCacheStrategy.PIPELINE,
    /**
     * Execution strategy for the tool calls within one round-trip. Default is sequential
     * (calls one by one through a channel), because builders in pipelines are not thread-safe
     * (see the KDoc of [ToolCallScheduler]). When the tools are completely independent,
     * it can be replaced with [ParallelToolCallScheduler].
     */
    private val toolCallScheduler: ToolCallScheduler = SequentialToolCallScheduler(),
    /**
     * Per-round-trip recovery (null = off). On a retryable failure of a single round-trip
     * (stream break, transient 5xx/429, malformed constrained-decoding response) the loop
     * does NOT abort the invocation: the failed attempt's output is discarded (it was never
     * appended to [historyPrompt]) and the loop re-attempts the SAME productive turn, which
     * re-renders the phase description with a FRESH state snapshot and re-prompts. The
     * conversation built by previous round-trips is preserved — recovery costs one
     * round-trip instead of replaying the whole invocation (which is what the outer
     * [org.nobirds.murmuration.agent.RetryAgent] does; it remains the fallback when this
     * budget is exhausted or the error is not retryable).
     *
     * Partial tool effects of the failed attempt are tolerated by design: tool calls that
     * were already received and executed keep their effects, and the next iteration's
     * snapshot shows the model the actual post-effect state, so it re-decides from reality
     * (same idempotency contract that full re-invocation retries rely on).
     *
     * The retry budget is **consecutive**: it resets after every successful round-trip, so
     * a long invocation may survive several isolated stream breaks while a persistently
     * failing round-trip still exhausts the budget quickly. A retry re-attempts the same
     * productive turn and does NOT consume the [maxTurns] budget — that budget bounds
     * *productive* round-trips only, retries are bounded solely by [RetryPolicy.maxRetries].
     * This matters for a failure on the very last productive turn: previously the scheduled
     * retry had no loop pass left to run in and the failure was silently swallowed as
     * "maxTurns exhausted" (neither retried in-loop nor propagated to the outer RetryAgent);
     * now the retry always gets its attempt regardless of turn position.
     */
    private val roundTripRetry: RetryPolicy? = null,
) : org.nobirds.murmuration.agent.WorkerAgent<ToolCallingInput<In, Tool>, Tool> {

    override val name: String = generateAgentName(name, "CallToolAgent")

    /**
     * Classifies a tool result as error/refusal. ALWAYS ON (there is no "off" — an unclassifiable
     * result simply counts as success), serving two consumers with one convention:
     *
     * 1. **Repeated-call de-duplication** (see [RepeatedCallTracker]): a stored error is never
     *    hard-skipped, and an identical repeat of an error gets the honest
     *    [REPEATED_CALL_REFUSAL_ADVISORY] instead of the "effect applied" [REPEATED_CALL_ADVISORY].
     * 2. **Forced error-visibility round-trip** — a framework-wide guard over the tool results of
     *    the LAST executed batch. It closes a blind spot of the state machine: when a batch both
     *    (a) produces a tool result the model never gets to react to and (b) moves the phase on
     *    (advancing it, or closing all tasks so the invocation finishes) on the SAME round-trip,
     *    any error hidden in that batch is swallowed — the model's own self-check ran blind and
     *    the only recovery is a costly downstream re-plan.
     *
     *    When the matcher flags at least one result of the batch, the framework HOLDS the
     *    between-round-trips transition apply for one forced round-trip: [Stateful.onAfterToolCalls]
     *    is skipped, so NO phase change lands, the error results (already appended to the
     *    conversation) are re-delivered, and the model gets one more turn IN THE CURRENT PHASE
     *    (its tools still available) to correct them before the state machine is allowed to move.
     *
     *    **The hold budget is per "place", not per invocation.** A "place" is the source phase plus
     *    the transition(s) the batch actually requested (a successful [Stateful.transitionTo] gate,
     *    see [TransitionResult.success] — a REFUSED transition registers nothing and is therefore
     *    not a place). The rule is «never twice in a row at the same place»: if the immediately
     *    preceding batch was already held for the same requested transition and the model produced
     *    an erroring batch requesting it again, the transition IS applied (with a WARN) — holding
     *    again could only loop. A different transition, or a transition from a different phase, is a
     *    different place and gets its own hold. This is what makes the guard work for MID-flow
     *    phase changes: an earlier errored batch elsewhere in the invocation can no longer eat the
     *    only hold and let a later `transitionTo` advance the phase over an unseen rejection (live
     *    incident 2026-07-30: a planner's `addDecisions` was rejected in two consecutive SETUP
     *    batches, the second of which also carried `transitionTo(ACTIONS)` — the phase advanced,
     *    the rejection was never seen, and the turn skeleton shipped empty).
     *
     *    A batch whose results error but which requests NO transition the framework can see (its
     *    phase change, if any, comes from the builder's internals — an armed auto-advance after the
     *    last task closed, an ON_FINDINGS self-review finish) keeps the original budget of exactly
     *    ONE hold per invocation. Such a hold cannot be attributed to a place, and an unbounded
     *    number of them would keep skipping [Stateful.onAfterToolCalls] — which is also where
     *    builders do their per-round-trip bookkeeping — for a whole invocation.
     *
     *    ⚠️ **The guard only works for DEFERRED appliers** (transitions landing in
     *    [Stateful.onAfterToolCalls], as the framework contract requires) — holding = skipping
     *    that apply. The framework's own fallback
     *    [org.nobirds.murmuration.tooling.defaultStateful] (every plain tool/classifier) IS such
     *    a deferred applier: an errored batch that also requested the terminal transition is
     *    genuinely held and re-prompted, and the deferred transition lands only after a clean
     *    round (previously it applied inline, the loop exited on the terminal node and the forced
     *    round-trip was silently annulled). For a USER [Stateful] that applies transitions
     *    IMMEDIATELY inside the tool call the hold still holds NOTHING, because the phase already
     *    flipped mid-batch — a SILENT HOLE: with MID-FLOW phases the forced round-trip is spent
     *    in the NEW phase, where the tools that would fix the error are no longer available; with
     *    a TERMINAL target the forced round-trip is annulled at the top of the loop and the error
     *    stays unseen. Live incident e814236c (2026-08-02, lite reply builder): a rejected
     *    `addDecision` + a `transitionTo(REVIEW)` in one batch — the hold fired, the phase
     *    advanced regardless, and the decision was lost silently. Such a hold is detected and
     *    WARNed below; the fix belongs in the builder (defer the apply, as the framework contract
     *    requires), not here.
     *
     * The matcher receives the rendered result text (a string primitive unquoted) for (2) and the
     * raw invocation result for (1). It is NOT configurable per agent: the matcher is always
     * [ErrorMatcher.resolveDefault], the single source shared with `TaskClosingDecorator` and the
     * instruments built by `buildInstruments` / `AgentToolMount.of`.
     */
    private val errorMatcher: ErrorMatcher get() = ErrorMatcher.resolveDefault()

    private val logger = logger()

    override suspend fun ExecutionContext.execute(input: ToolCallingInput<In, Tool>): Tool {
        val displayName = getDisplayName(name)
        logger.info("Agent {} started", displayName)

        val startTime = TimeSource.Monotonic.markNow()
        // Set once a failing round-trip has already reported its own FAILURE entry (keyed by its
        // executionId, which closes its RUNNING row). Guards the catch below from writing a
        // second, orphaned FAILURE row for the very same failure.
        var roundTripFailureReported = false
        try {
            val prompt = with(promptGenerator) { generate(input) }
            val effectiveMaxTurns = maxTurnsFn?.invoke(input.input) ?: maxTurns
            executeWithTools(displayName, prompt, input.tool, effectiveMaxTurns) { roundTripFailureReported = true }
            // No aggregate report here: per-iteration entries already cover token usage and
            // duration. Reporting a summed-up totalStats on top would make LlmCostCalculator
            // count every iteration twice.
            logger.info("Agent {} finished in {}", displayName, startTime.elapsedNow())

            return input.tool
        } catch (e: CancellationException) {
            // cooperative cancellation is never recovered from and
            // never gets a FAILURE stub — `executeWithToolsInner`'s own `catch (CancellationException)`
            // (below) already reported `onFailure` and rethrew WITHOUT calling `markRoundTripReported`,
            // so `roundTripFailureReported` stays `false` here. Without this catch, the generic
            // `catch (Exception)` below inherits the rethrow (`kotlinx.coroutines.CancellationException`
            // is `java.util.concurrent.CancellationException`, itself an `IllegalStateException`, hence
            // an `Exception`) and — seeing `roundTripFailureReported == false` — writes a SECOND,
            // fabricated CLOSED record with a fresh random `executionId`, alongside the RUNNING row the
            // cancelled round-trip already left open (cancellation leaves exactly ONE
            // gap, the RUNNING row — not two artifacts). Rethrown untouched, no report written here.
            throw e
        } catch (e: Exception) {
            val duration = startTime.elapsedNow()
            // Failure stub covering exceptions that killed the loop with no round-trip of its own
            // to report: prompt generation, the first checkCallLimit, a CallLimitDecision.Throw
            // raised before onAgentTurn, or post-success processing. When a round-trip itself
            // failed it already reported a FAILURE entry keyed by its executionId (closing its
            // RUNNING row), so writing another one here would just duplicate it — skip.
            if (!roundTripFailureReported) {
                val typed = e as? PromptExecutionException
                // Whatever partial statistics a nested round-trip's own assembly already
                // attached to this exception before it reached here (tokens, responseMessage)
                // — this stub covers failures with NO round-trip of its own to
                // report, so there usually is none, but it is not thrown away when there is.
                val partial = typed?.partialStats
                val stats = (partial ?: PromptExecutionStatistics()).copy(
                    success = false,
                    errorMessage = e.message,
                    attempts = 1,
                    errorCode = typed?.errorCode(),
                    // `?:` — do not clobber a non-null
                    // `partial.finishReason`/`streamDiagnostics` with the null
                    // `finishReasonOrNull()`/`streamDiagnostics` a non-`PolicyViolationException`
                    // always produces here.
                    finishReason = typed?.finishReasonOrNull() ?: partial?.finishReason,
                    streamDiagnostics = typed?.streamDiagnostics ?: partial?.streamDiagnostics,
                )
                reportAgentExecution(PromptExecutionEntry(displayName, duration, stats))
            }
            logger.warn("Agent {} failed in {}. Error: {}", displayName, duration, e.message)
            throw e
        }
    }

    private suspend fun ExecutionContext.executeWithTools(displayName: String, prompt: Prompt, tool: Tool, maxTurns: Int, markRoundTripReported: () -> Unit): PromptExecutionStatistics {
        @Suppress("UNCHECKED_CAST")
        val stateful = AgentToolMount.statefulOf(tool) as Stateful<AgentState, Tool>

        @Suppress("UNCHECKED_CAST")
        val stateMachine = stateful.stateMachine as StateMachineDefinition<AgentState, Any>

        // Self-listener auto-subscription: a tool/stateful that implements AgentEventListener
        // is subscribed for the duration of this call and unsubscribed in `finally`. The
        // listener passed to the agent constructor goes through the same path. All of them
        // become visible to every `dispatchToolCalling` invocation while the agent runs.
        val toolHandle = (tool as? AgentEventListener)?.let { register(it) }
        val statefulHandle = if (stateful !== tool) (stateful as? AgentEventListener)?.let { register(it) } else null
        val externalHandle = eventListener?.let { register(it) }
        try {
            return executeWithToolsInner(displayName, prompt, tool, stateful, stateMachine, maxTurns, markRoundTripReported)
        } finally {
            externalHandle?.close()
            statefulHandle?.close()
            toolHandle?.close()
        }
    }

    private suspend fun ExecutionContext.executeWithToolsInner(
        displayName: String,
        prompt: Prompt,
        tool: Tool,
        stateful: Stateful<AgentState, Tool>,
        stateMachine: StateMachineDefinition<AgentState, Any>,
        maxTurns: Int,
        markRoundTripReported: () -> Unit,
    ): PromptExecutionStatistics {
        // Define all tools once at the beginning: the single writer of "what is mounted".
        val mount = AgentToolMount.of(tool, stateful)

        // Global system instructions - generated once to stay stable in the prompt prefix
        val globalSystemInstruction = stateMachine.generateGlobalInstruction(stateful)

        // Flatten all instruments and apply `Stateful.defaultUserToolPhases` so user-defined
        // setters get a phase restriction without needing `@Phases` on every method. Used by
        // [appendPhaseDescription] to render the per-phase allowlist into the user-block.
        val allInstruments = mount.instruments

        // `historyPrompt` is the append-only conversation: prefix + per-round-trip phase
        // description + accumulated model/tool content. It only grows, never mutates earlier
        // entries — so the cacheable prefix on Vertex implicit cache keeps growing with it.
        //
        // Tool gating per phase is enforced by the runtime guard inside [ToolsExecutor.executeCall]:
        // calls to instruments whose `phases` do not include the current state get a synchronous
        // "Error: ..." result instead of executing. Provider-level allowlists (Gemini
        // `allowedFunctionNames`) are not used — they are incompatible with cached content
        // (Vertex returns 400 if `tools/tool_config` are set on requests that reuse a cache).
        // The model is told which tools are allowed via a text block emitted by
        // [appendPhaseDescription] each round-trip.
        // The framework SYSTEM block is INSERTED right after the generator's own SYSTEM
        // contents instead of being appended at the tail. Appending used to place this
        // stable block AFTER mutating user contents (e.g. per-run history items), so the
        // cacheable content list of run N was never a prefix of run N+1 — cross-pipeline
        // (TTL_BASED) prefix caching could never match. The relative order of SYSTEM
        // contents is preserved, and providers merge SYSTEM parts into a single system
        // instruction by that relative order — the model sees an identical prompt; only
        // the prefix/hash structure changes.
        val frameworkSystemContent = PromptContentBuilder(PromptContentType.SYSTEM).apply {
            text("CRITICAL: You are a tool-calling agent. You MUST use the provided tools for ALL operations and changes. Do not respond with plain text. When you have completed all necessary actions, you MUST call the appropriate completion tool.", cacheStrategy = frameworkInstructionsCacheStrategy)
            // Add global state machine instructions once
            text(globalSystemInstruction, label = "STATE MACHINE FLOW", cacheStrategy = frameworkInstructionsCacheStrategy)
        }.build()
        var historyPrompt = prompt.update {
            settings = mount.toolsSettings(prompt.settings)
        }.let { p ->
            val insertAt = p.content.indexOfLast { it.type == PromptContentType.SYSTEM } + 1
            Prompt(p.llm, p.settings, p.content.toMutableList().apply { add(insertAt, frameworkSystemContent) }.toList())
        }

        // Marks on the history blocks the agent appends itself (phase blocks, deferred-transition notice): the same strategy as
        // the exchange history, derived from the model's cache kind. On MARKS they are as stable as the history once appended
        // (nothing prunes them), so a NO_CACHE block here would be a hole that ends the cacheable prefix before the tail.
        val historyCache = CachePlacement.of(prompt.llm.cacheKind).historyCacheStrategy

        var totalStats = PromptExecutionStatistics()

        // Per-round-trip recovery state (see [roundTripRetry]): created lazily on the first
        // failure, dropped after every successful round-trip — the budget is consecutive.
        var retrySession: RetrySession? = null

        // Memory of every executed (name,args)→result across ALL round-trips of THIS invocation.
        // Consulted inside [ToolsExecutor.executeCall] to soft-flag the first mechanical no-op
        // repeat and HARD-SKIP subsequent ones over the whole run (not just against the previous
        // round-trip, which alternating stalls slip through) — see [RepeatedCallTracker].
        val repeatedCalls = RepeatedCallTracker()

        // `productiveTurns` bounds *productive* round-trips (a turn that reached the model and
        // was processed, success or graceful stop) — it is the [maxTurns] budget. A retryable
        // failure re-attempts the same productive turn WITHOUT advancing this counter, so a
        // retry never eats into the productive budget (and a failure on the last turn is no
        // longer swallowed by loop exhaustion). `attemptNo` is a monotonic naming counter that
        // increments on every pass (retries included) so each attempt gets a distinct
        // iteration name / RUNNING row.
        var productiveTurns = 0
        var attemptNo = 0
        // [forced error-visibility] The "place" (source phase + requested transition) the
        // IMMEDIATELY PRECEDING batch was held at, or null when that batch was not held. Holding
        // twice in a row at the same place could only loop, so a repeat is applied with a WARN
        // instead. See [errorMatcher].
        var lastHoldPlace: String? = null
        // [forced error-visibility] Spent once an errored batch that requested NO framework-visible
        // transition has been held this invocation. Such a hold has no place to be keyed on, so it
        // keeps the original budget of one per invocation. See [errorMatcher].
        var placelessHoldUsed = false
        while (productiveTurns < maxTurns) {
            val startTime = TimeSource.Monotonic.markNow()

            val currentState = stateful.currentState
            val node = stateMachine.getNode(currentState)

            // Bake the current state into the iteration name so logs and `agent_name` in DB
            // expose the phase at-a-glance:
            //   Agent#abcd#1[PLANNING][CallToolAgent]
            val iterationName = displayName.withIteration(++attemptNo).withState(currentState)
            logger.debug("CallingAgent {} started", iterationName)

            if (node is StateMachineNode.Terminal) {
                logger.debug("CallingAgent {} reached terminal state: {}", iterationName, currentState)
                return totalStats
            }

            // Snapshot the state *before* the round-trip so the safety-net auto-transition
            // doesn't swallow a transition that happened inside the same round-trip (the
            // model wouldn't have seen the new phase instruction yet).
            val stateBeforeRoundTrip = currentState

            stateful.onRoundTripStart(stateBeforeRoundTrip)
            dispatchToolCalling { onBeforeTurn(stateBeforeRoundTrip) }

            // Append the current phase description to history *every* round-trip. The
            // description text is identical across round-trips inside the same phase, so it
            // becomes part of the stable, cacheable prefix on subsequent turns. After a
            // transition the new phase description gets appended on the very next round-trip.
            historyPrompt = historyPrompt.appendPhaseDescription(stateMachine, currentState, stateful, allInstruments, historyCache)

            val agentPrompt = historyPrompt
            val settings = agentPrompt.settings as? ToolsPromptSettings
                ?: throw PromptExecutorException("Prompt settings must be ToolsPromptSettings")

            when (val decision = beforeCall()) {
                is CallLimitDecision.Proceed -> Unit
                is CallLimitDecision.GracefulStop -> {
                    logger.warn("Agent {} call limit exhausted: {}. Returning partial result.", iterationName, decision.reason)
                    // Loop abandons any still-open work here — give the builder its salvage hook
                    // (e.g. escalate uncompleted tasks) BEFORE the observation listeners fire.
                    stateful.onForcedTermination(currentState, decision.reason)
                    dispatchToolCalling { onCallLimitExhausted(currentState, decision.reason) }
                    return totalStats
                }
                is CallLimitDecision.Throw -> throw decision.cause
            }

            // Per-round-trip correlation id: paired with the PromptExecutionEntry reported on
            // completion, so a durable writer can INSERT a RUNNING row here and UPDATE it there.
            val executionId = UUID.randomUUID()
            onAgentTurn(executionId, displayName)

            // On-the-fly tool execution: tool jobs start the moment a `ToolCallReceived` arrives,
            // in parallel with the model still generating. A broken stream (Failed / exception)
            // does NOT cancel them: `executeTurn`'s `finally` AWAITS every launched job before
            // closing the session (see the `toolJobs.forEach { it.await() }` there), so tools
            // already dispatched in this round-trip run to completion and their mutations land in
            // the builder even when the round-trip itself fails.
            val toolsExecutor = ToolsExecutor(iterationName, executionId, mount.instruments, stateful, this, repeatedCalls)
            val turnResult = try {
                // Publish this invocation's execution identity (pipelineId + agent displayName)
                // into the coroutine context for executors that support identity-aware request
                // handling (no-op on providers without one) — see [AgentExecutionIdentity] KDoc.
                // `displayName` (not `iterationName`) is used so the identity stays the SAME
                // across every round-trip of this invocation — exactly the case a provider's
                // routing hint exists for: a multi-round tool-calling loop whose growing prefix
                // should keep landing on the same warm replica. Must wrap the round-trip's actual
                // `.collect` (inside `executeTurn`) — wrapping prompt construction alone would
                // have no effect, since `PromptExecutor.execute` returns a cold `Flow`.
                withContext(AgentExecutionIdentity(pipelineId, displayName, stepPath)) {
                    toolsExecutor.executeTurn(agentPrompt)
                }
            } catch (e: CancellationException) {
                // Cooperative cancellation is never recovered from.
                onFailure(e)
                throw e
            } catch (e: Throwable) {
                val session = retrySession
                    ?: roundTripRetry?.newSession()?.also { retrySession = it }
                val retryDelay = session?.nextDelayOrNull(e)
                // Close THIS iteration's RUNNING row (opened by onAgentTurn above) with real
                // failure diagnostics instead of leaving it for the blunt end-of-pipeline orphan
                // sweep: error code/message, stream diag, and whatever the round-trip had already
                // assembled before it broke — tokens, responseMessage (tool_call parts included),
                // model name. Keyed by executionId →
                // UPDATEs that exact row. A retried attempt thus lands as its own FAILURE row,
                // making in-loop retries visible in the DB.
                //
                // `partial` covers the OBRYV-STREAM path (`executeTurn`'s own `Failed`-event
                // throw, which assembles `stats` — including `responseMessage` — before
                // rethrowing): it is null on the OTHER failure path this catch also handles, a
                // throw out of a tool-job (e.g. "Unknown tool: …" — `ToolsExecutor.executeCall`),
                // which blows up `coroutineScope` before `executeTurn` ever assembles anything.
                // Either way `toolOutcomes` below is NOT read from `partial`/`responseMessage` —
                // it comes from `toolsExecutor`'s own accumulator, which is the only thing that
                // survives BOTH failure paths.
                val typed = e as? PromptExecutionException
                val partial = typed?.partialStats
                val failStats = (partial ?: PromptExecutionStatistics()).copy(
                    // The stream may have broken before (or without) reporting a `PromptStarted`
                    // event, so the actual API-returned model name can be empty even when
                    // `partial` is non-null. Fall back to the configured `Llm.model` — same
                    // source `CacheManager` uses for `requestedModelName` — so
                    // `LlmCostCalculator` (which groups by `modelName` and skips empty ones) and
                    // admin's byModel breakdown still attribute this failure's partial tokens to
                    // a model instead of dropping them silently.
                    modelName = partial?.modelName?.takeIf { it.isNotBlank() } ?: agentPrompt.llm.model,
                    requestedModelName = partial?.requestedModelName ?: agentPrompt.llm.model,
                    success = false,
                    errorMessage = e.message,
                    attempts = 1,
                    errorCode = typed?.errorCode(),
                    // `?:` — not an unconditional overwrite: `typed?.
                    // finishReasonOrNull()` is non-null ONLY for `PolicyViolationException`, so on
                    // every other exception type this would otherwise CLOBBER a value `partial`
                    // already carried (e.g. `MaxTokensException` breaking a round-trip whose
                    // `stats.finishReason` was already set to "MAX_TOKENS" before the throw).
                    finishReason = typed?.finishReasonOrNull() ?: partial?.finishReason,
                    streamDiagnostics = typed?.streamDiagnostics ?: partial?.streamDiagnostics,
                )
                reportAgentExecution(
                    PromptExecutionEntry(
                        iterationName, startTime.elapsedNow(), failStats,
                        executionId = executionId,
                        toolOutcomes = toolsExecutor.outcomesSoFar(),
                    )
                )
                if (retryDelay == null) {
                    markRoundTripReported()
                    onFailure(e)
                    throw e
                }
                // Round-trip recovery: drop the failed attempt (nothing of it was appended
                // to the conversation), pause, and re-attempt the SAME productive turn — the
                // next pass re-renders the phase description with a fresh state snapshot and
                // re-prompts on the preserved history. `continue` does NOT advance
                // `productiveTurns`, so the retry is bounded by [RetryPolicy.maxRetries], not
                // by the productive-turn budget (a retry on the last turn is no longer lost).
                // See [roundTripRetry].
                // Salvage hook BEFORE the replay: tool jobs of the failed attempt already ran to
                // completion (see `executeTurn`'s `finally`), while `continue` skips
                // [Stateful.onAfterToolCalls] — so whatever that batch buffered for the END of the
                // round-trip would otherwise survive into the retried turn and get executed for a
                // batch the model is about to re-issue (duplicate/ghost work). See
                // [Stateful.onRoundTripAborted].
                //
                // The hook runs while we are ALREADY handling a recoverable failure — it must not
                // be allowed to escalate that into an unrecoverable one. A throwing salvage hook
                // is a bug in the hook, not a reason to abandon a retry the framework judged safe.
                // Cooperative cancellation is never recovered from (same policy as the outer catch
                // above), so it is rethrown rather than logged-and-swallowed.
                try {
                    stateful.onRoundTripAborted(stateBeforeRoundTrip, e)
                } catch (hookError: CancellationException) {
                    throw hookError
                } catch (hookError: Throwable) {
                    // An abort from the hook leaves the loop: no retry. The FAILURE row of this
                    // round-trip is already written, so mark it reported (otherwise the outer
                    // catch writes a second stub row) and fire `onFailure`, exactly as the
                    // non-retryable branch does. No retry event was sent: there is no retry.
                    if (hookError.abortsAgent) {
                        markRoundTripReported()
                        onFailure(hookError)
                    }
                    hookError.rethrowIfAborts()
                    logger.warn(
                        "Agent {} onRoundTripAborted salvage hook threw during retry handling — " +
                            "ignoring, retry proceeds regardless: {}",
                        iterationName, hookError.message, hookError,
                    )
                }
                // The retry is announced only once the hook let it proceed.
                reportAgentRetry(iterationName, session.attempt, e)
                logger.warn(
                    "Agent {} round-trip failed (retry {}/{}), retrying in {}: {}",
                    iterationName, session.attemptInBudget, session.maxRetries, retryDelay, e.message,
                )
                delay(retryDelay)
                continue
            }
            // A successful round-trip resets the consecutive recovery budget.
            retrySession = null

            val duration = startTime.elapsedNow()
            val iterationStats = turnResult.stats

            totalStats = totalStats.mergeWith(iterationStats)

            reportAgentExecution(
                PromptExecutionEntry(
                    iterationName, duration, iterationStats,
                    executionId = executionId,
                    toolOutcomes = turnResult.toolOutcomes,
                )
            )

            afterRequest(toolCallsInResponse = turnResult.toolCalls.size)

            // Repeated no-op calls are already flagged (advisory) / hard-skipped (NO-OP) inside
            // [ToolsExecutor.executeCall] via [repeatedCalls], so the tool results committed to
            // history here already carry their annotations — see [RepeatedCallTracker].
            historyPrompt += turnResult.asContent(iterationName)

            // [forced error-visibility] Did the batch we just committed to history carry a result
            // the [errorMatcher] classifies as an error? Computed BEFORE the transition apply
            // below, because the whole point is to intercept a batch that would otherwise close
            // the phase / finish the invocation with an unseen error inside it. The RENDERED text
            // is what is matched (a string primitive unquoted) — `JsonElement.toString()` would
            // wrap it in quotes and defeat every prefix convention.
            val errorInLastBatch = turnResult.toolResults.any { errorMatcher.isError(it.result.renderText()) }

            // [forced error-visibility] WHERE would this batch have taken the state machine? The
            // executor collected the transition(s) this batch successfully REQUESTED (a refused
            // `transitionTo` registers nothing — see [ToolsExecutor.requestedTransitions]); together
            // with the phase the round-trip started in they identify the "place" a hold happens at.
            // Null when the batch requested no transition the framework can see: its phase change,
            // if any, lives inside the builder (armed auto-advance, self-review finish) and gets the
            // single placeless hold instead.
            val requestedTransitions = toolsExecutor.requestedTransitions()
            val transitionPlace = requestedTransitions
                .takeIf { it.isNotEmpty() }
                ?.let { "$stateBeforeRoundTrip=>${it.joinToString(",")}" }
            val holdPlace = transitionPlace ?: PLACELESS_HOLD

            // HOLD the between-round-trips transition apply for one turn: skip
            // [Stateful.onAfterToolCalls] entirely (so no phase advance or terminal transition
            // lands) and re-prompt. The error results were just appended to the conversation above,
            // so the model sees them on the very next round-trip and can correct them IN THE CURRENT
            // PHASE (its tools are still available — the phase never advanced). The held round-trip
            // still counts against the [maxTurns] budget. Budget rules — see [errorMatcher]:
            // never twice in a row at the same place, and one placeless hold per invocation.
            val holdThisBatch = when {
                !errorInLastBatch -> false
                holdPlace == lastHoldPlace -> false
                transitionPlace == null -> !placelessHoldUsed
                else -> true
            }
            if (holdThisBatch) {
                lastHoldPlace = holdPlace
                if (transitionPlace == null) placelessHoldUsed = true
                logger.info(
                    "Agent {} last batch produced error tool-result(s) (requested transitions: {}); forcing one extra round-trip so the model can see and fix them before the phase advances/finishes",
                    iterationName, requestedTransitions.ifEmpty { listOf("none") },
                )
                // INERT HOLD: the Stateful applied its transition INSIDE the batch, so skipping
                // `onAfterToolCalls` held nothing (see [errorMatcher]). Non-terminal target: the
                // forced round-trip runs in the NEW phase, where the tools that would fix the error
                // may be gone. Terminal target: the next loop pass lands on the terminal node and
                // returns — the forced round-trip is annulled outright and the error stays unseen.
                val stateAfterBatch = stateful.currentState
                if (stateAfterBatch != stateBeforeRoundTrip) {
                    logger.warn(
                        "Agent {} hold is INEFFECTIVE: the phase already moved {} -> {} INSIDE the errored batch " +
                            "(immediate-applier Stateful). {} — the builder must defer its transition apply to onAfterToolCalls.",
                        iterationName, stateBeforeRoundTrip, stateAfterBatch,
                        if (stateMachine.getNode(stateAfterBatch) is StateMachineNode.Terminal)
                            "The new phase is TERMINAL, so the forced round-trip is annulled at the top of the loop and the error stays unseen"
                        else
                            "The forced round-trip will run in the new phase and the error may be unfixable there",
                    )
                } else if (requestedTransitions.isNotEmpty()) {
                    // The batch's transitionTo gate already answered "Success" — tell the model that
                    // the accepted transition is DEFERRED and it is still in the source phase, so the
                    // forced round-trip is spent fixing the errors instead of chasing a phantom phase
                    // change (deferred appliers re-apply after a clean round; re-issuing the call is
                    // always safe).
                    historyPrompt = historyPrompt.update {
                        user {
                            text(
                                "The transition you requested (${requestedTransitions.joinToString(", ")}) was accepted but is " +
                                    "DEFERRED — NOT applied yet, because this batch produced error tool-result(s). You are still " +
                                    "in phase '$stateBeforeRoundTrip' with its tools available. Fix the errors above first: the " +
                                    "transition applies only after a clean round (you may re-issue transitionTo together with the fixes).",
                                label = "TRANSITION DEFERRED",
                                cacheStrategy = historyCache,
                            )
                        }
                    }
                }
                dispatchToolCalling { onTurnFinished(stateful.currentState) }
                productiveTurns++
                continue
            }
            if (errorInLastBatch) {
                logger.warn(
                    "Agent {} letting the transition apply run with error tool-result(s) still present (place: {}); not holding again to avoid a livelock",
                    iterationName, holdPlace,
                )
            }
            lastHoldPlace = null

            // Single apply point for state transitions (they land BETWEEN round-trips, see
            // [Stateful.transitionTo] / [Stateful.onAfterToolCalls]): an explicitly requested
            // transition, an armed auto-advance, or a silent self-review finish is resolved
            // here. A transition's effect can be heavy — a planner's resolve-on-transition
            // runs a whole wave of sub-agents inside the effect — so a failure here must NOT
            // kill the invocation. We restore the pre-deferral fault-tolerance (back then the
            // effect ran inside the tool call and a throw was caught by [ToolsExecutor.executeCall],
            // surfacing as an "Error:" tool result while the loop continued): on a non-fatal
            // failure the transition is simply not applied, the loop continues, and the Stateful
            // is expected to have recorded the reason for the model to see next round-trip (an
            // application typically does this via its transition-cancelled note). Cooperative cancellation and
            // fatal Errors are never swallowed. Any partially-applied effect (a wave that died
            // mid-flight) leaves dirty state — a pre-existing property tolerated by idempotent,
            // registry-backed tools — but the state transition itself does not apply.
            try {
                stateful.onAfterToolCalls(stateBeforeRoundTrip)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Error) {
                throw e
            } catch (e: Exception) {
                // An abort (bare or wrapped) is not a "transition apply failure": it cancels the
                // whole run and must leave the loop as it came.
                e.rethrowIfAborts()
                logger.warn(
                    "Agent {} transition apply failed; transition NOT applied, continuing loop: {}",
                    iterationName, e.message, e,
                )
            }

            val nextState = stateful.currentState
            val nextNode = stateMachine.getNode(nextState)

            // Observability of the applied transition. Detected by diffing the state the
            // round-trip started in against the state after the apply point — this covers both
            // deferred appliers (state flips here; the framework's own defaultStateful included)
            // and user Statefuls that still apply inside a tool call, so the per-tool-call diff
            // that used to live in [ToolsExecutor.executeCall] is no longer needed.
            if (nextState != stateBeforeRoundTrip) {
                dispatchToolCalling { onStateTransition(stateBeforeRoundTrip, nextState) }
            }

            dispatchToolCalling { onTurnFinished(nextState) }

            if (nextNode is StateMachineNode.Terminal) {
                // The "finishing with unfixed errors" case is already WARNed at the hold decision
                // above (it is the same fall-through), so no second warning here.
                logger.debug("Agent {} finished via auto-transition to terminal state: {}", iterationName, nextState)
                return totalStats
            }

            // A productive turn completed (reached the model and was processed) — only these
            // count against the [maxTurns] budget. Retries `continue` above without reaching here.
            productiveTurns++
        }

        val finalState = stateful.currentState

        logger.warn("Agent {} exhausted maxTurns={} without reaching terminal state (current={}). Returning partial result; verifier next round will pick up unfinished work.", displayName, maxTurns, finalState)

        // Same salvage hook as the GracefulStop path: the loop is abandoning open work, let the
        // builder escalate it before the observation listeners see the exhaustion.
        stateful.onForcedTermination(finalState, "maxTurns=$maxTurns exhausted")
        dispatchToolCalling { onMaxTurnsExhausted(finalState) }

        return totalStats
    }

    private fun Prompt.appendPhaseDescription(
        stateMachine: StateMachineDefinition<AgentState, Any>,
        state: AgentState,
        stateful: Stateful<AgentState, *>,
        allInstruments: List<InstrumentDeclaration<*>>,
        historyCache: PromptCacheStrategy,
    ): Prompt {
        val node = stateMachine.getNode(state)
        if (node is StateMachineNode.Terminal) return this
        val description = stateMachine.generateCurrentStateInstruction(state, stateful)
        val phaseName = state.toString()
        val staticallyAllowed = allInstruments
            .filter { it.phases.isEmpty() || phaseName in it.phases }
            .map { it.name }
            .distinct()
        // Dynamic substate narrowing on TOP of the static @Phases gate (see
        // [Stateful.availableTools]): the builder may collapse the allowlist further given its
        // live substate. Filter (not replace) so the advertised order stays deterministic.
        val allowedNow = stateful.availableTools(state, staticallyAllowed.toSet())
        val available = staticallyAllowed.filter { it in allowedNow }
        val toolsBlock = buildString {
            append("Tools available in phase '").append(phaseName).append("': ")
            append(available.joinToString(", "))
            append(". Calling any other tool returns an error.")
        }
        val snapshot = stateful.currentReplaceableSnapshot()?.takeIf { it.text.isNotBlank() }
        return update {
            user {
                text(
                    description,
                    label = "PHASE $state",
                    cacheStrategy = historyCache,
                    replaceableSectionId = PHASE_DESCRIPTION_SECTION,
                )
                text(
                    toolsBlock,
                    label = "PHASE TOOLS",
                    cacheStrategy = historyCache,
                    replaceableSectionId = PHASE_DESCRIPTION_SECTION,
                )
                if (snapshot != null) {
                    text(
                        snapshot.text,
                        label = "CURRENT OUTPUT SNAPSHOT",
                        cacheStrategy = historyCache,
                        replaceableSectionId = snapshot.sectionId,
                    )
                }
            }
        }
    }

    private fun TurnResult.asContent(iterationName: String): List<PromptContent> = if (toolCalls.isNotEmpty()) {
        logger.debug("Agent {} finished with {} tool calls", iterationName, toolCalls.size)

        // Append model response + tool results to the append-only history.
        asContent()
    } else {
        val responseText = textParts.asSequence()
            .filterNot { it.isThought }
            .joinToString("") { it.text }

        logger.debug("Agent {} finished with text response: {}", iterationName, responseText)

        // Each textPart (including chain-of-thought blocks with their signatures) goes into
        // its own TextPromptPart — the structure of the model's answer is preserved without gluing,
        // and the signatures of all thought blocks travel into the follow-up request correctly.
        val modelParts = textParts.map { part ->
            TextPromptPart(
                part.text,
                part.thoughtSignature,
                cacheStrategy = CachePlacement.of(llm.cacheKind).historyCacheStrategy,
                kind = part.promptKind(),
            )
        }

        listOf(PromptContent(PromptContentType.MODEL, modelParts))
    }

    private suspend fun ToolsExecutor.executeTurn(agentPrompt: Prompt): TurnResult = coroutineScope {
        val collectedParts = mutableListOf<MessagePart>()
        // The agent's history of the turn: only text, thought and call parts, in arrival order
        // (InlineData/File/Embedding never enter the history). The single source of the turn.
        val historyParts = mutableListOf<MessagePart>()

        // Tool-call session per round-trip. Scheduler (default — sequential, chain of
        // Deferreds) decides whether to run the calls one by one or in parallel. Sequential is the
        // safe default: builders in pipelines mutate shared state through
        // `shared.X = shared.X.copy(...)` (two parallel read-modify-writes overwrite
        // each other — bug 2026-05-19, addClass lost classes because of updateBasicInfo).
        // The scheduler wraps ToolsExecutor (it implements ToolExecutor). A session
        // has the same `execute(call)` interface plus close().
        val toolSession = toolCallScheduler.open(this, this@executeTurn)
        val toolJobs = mutableListOf<Deferred<ToolOutcome>>()

        var started: PromptStarted? = null
        var servedModelName: String? = null
        var inputTotal = org.nobirds.murmuration.prompt.InputTokenStatistics.EMPTY
        var outputTotal = OutputTokenStatistics.EMPTY
        var finishReason: String? = null
        var success = true
        var errorMessage: String? = null
        var collectedFailure: Throwable? = null
        var streamDiagnostics: StreamDiagnostics? = null

        // Single assembly point of the round-trip's statistics: the normal path and the failure
        // path (`catch` below) both go through it, so a stream that dies in any way — `Failed`
        // event, a throw, or both (the real Gemini shape: `emit(Failed)` then `throw`) —
        // carries the same partial content out. The field mapping lives in the core
        // `buildStatistics`, shared with `aggregate`.
        fun currentStatistics(): PromptExecutionStatistics = buildStatistics(
            started = started,
            servedModelName = servedModelName,
            input = inputTotal,
            outputTotal = outputTotal,
            responseMessage = if (collectedParts.isEmpty()) null
            else PromptMessage(role = PromptContentType.MODEL, parts = collectedParts.toList()),
            success = success,
            errorMessage = errorMessage,
            finishReason = finishReason,
            streamDiagnostics = streamDiagnostics,
        )

        try {
            promptExecutor.execute(agentPrompt).collect { event ->
                when (event) {
                    is PromptStarted -> started = event
                    is StreamResponded -> {
                        val requested = checkNotNull(started) { "StreamResponded arrived before PromptStarted" }
                        servedModelName = event.servedModelName
                        reportStreamStarted(event.servedModelName.ifEmpty { requested.modelName })
                    }
                    is PartReceived -> {
                        val part = event.part
                        collectedParts += part
                        when (part) {
                            is MessagePart.Text -> historyParts += part
                            is MessagePart.Thought -> historyParts += part
                            is MessagePart.ToolCall -> {
                                val call = part.toToolCall()
                                historyParts += part
                                // async wrapper on the outside: execute is suspending, and in sequential
                                // it waits its turn in the channel — without blocking us. This way
                                // we keep collecting events from the LLM stream while
                                // the previous tool calls are being executed.
                                toolJobs += async { toolSession.execute(call) }
                            }
                            is MessagePart.InlineData, is MessagePart.File, is MessagePart.ToolResult, is MessagePart.Embedding -> Unit
                        }
                    }
                    is UsageDelta -> {
                        inputTotal += event.input
                        outputTotal += event.output
                    }
                    is Completed -> {
                        finishReason = event.finishReason
                        streamDiagnostics = event.streamDiagnostics
                    }
                    is Failed -> {
                        success = false
                        errorMessage = event.error.message
                        finishReason = event.finishReason
                        streamDiagnostics = event.streamDiagnostics
                        collectedFailure = event.error
                    }
                    is CacheCreated -> reportCacheCreated(event)
                }
            }
        } catch (e: PromptExecutionException) {
            // Failure path: a provider that throws (with or without a preceding `Failed` event)
            // bypasses the assembly after this try. Assemble here so `partialStats` survives
            // every way a stream can die — only when `PromptStarted` (which precedes the
            // request) arrived; before that there is genuinely nothing to carry.
            if (success) {
                success = false
                errorMessage = e.message
            }
            if (started != null) e.partialStats = currentStatistics()
            throw e
        } finally {
            // Wait for the pending async tool jobs BEFORE closing the session. async {} starts
            // with a delay, and in kotlinx-coroutines 1.11.0+ its scheduling changed so that
            // `channel.send(...)` inside `toolSession.execute(call)` may run
            // AFTER `toolSession.close()` — this causes ClosedSendChannelException and breaks
            // tests with mocks. runCatching swallows the errors so that the main processing
            // (collectedFailure / TurnResult below) sees a meaningful failure cause.
            // An abort from a tool-job outranks the stream's own failure: remember it, still
            // await every job and close the session, then rethrow it from here (replacing the
            // stream error that `collectedFailure` would throw below).
            var toolJobAbort: Throwable? = null
            toolJobs.forEach { job ->
                runCatching { job.await() }.onFailure { if (toolJobAbort == null && it.abortsAgent) toolJobAbort = it }
            }
            // The stream is over (or failed) — close the session so the collector drains
            // the queue and finishes. Without close() it would hang in `for(channel)` forever and
            // coroutineScope would hang.
            toolSession.close()
            toolJobAbort?.rethrowIfAborts()
        }

        // Assembly moved ABOVE the `collectedFailure` throw below: on the
        // obryv-stream path (a `Failed` event, handled here — as opposed to a throw out of a
        // tool-job, which blows up `coroutineScope` before this point is ever reached) the
        // round-trip's `responseMessage` and tokens are fully known by now, and the FAILURE
        // record the loop's catch builds from the rethrown exception must not lose them.
        val s = started
        val stats = currentStatistics()

        collectedFailure?.let { f ->
            // Carry the fully-assembled `stats` out through the exception — including
            // `responseMessage` (tool_call parts already received survive here) and tokens — so
            // the FAILURE record rebuilt by the loop's catch (and the outer agent catch) reflects
            // the round-trip's real partial content instead of an empty stub. Only
            // when at least `PromptStarted` arrived — keep `partialStats` null when there is
            // genuinely nothing to carry (the stream broke before its first event).
            if (s != null) {
                (f as? PromptExecutionException)?.partialStats = stats
            }
            throw f
        }

        // Sole projection point: ToolOutcome -> ToolResultPromptPart happens HERE, on batch
        // assembly, and nowhere inside executeCall's four returns (see ToolOutcome KDoc).
        val outcomes = toolJobs.awaitAll()
        TurnResult(
            parts = historyParts.toList(),
            toolResults = outcomes.map { it.toPromptPart() },
            toolOutcomes = outcomes,
            stats = stats,
            toolsExecutor = this@executeTurn,
            llm = agentPrompt.llm,
        )
    }
}

private class ToolsExecutor(
    private val iterationName: String,
    private val executionId: UUID,
    mountedInstruments: List<InstrumentDeclaration<*>>,
    private val stateful: Stateful<AgentState, *>,
    private val ec: ExecutionContext,
    private val repeatedCalls: RepeatedCallTracker,
) : ToolExecutor {

    /**
     * The single entry point through which any scheduler ([SequentialToolCallScheduler],
     * [ParallelToolCallScheduler]) drives this executor — [executeCall] itself is `private`
     * precisely so this stays true BY CONSTRUCTION rather than by grep:
     * recording into [collectedOutcomes] happens HERE, once, so a second internal call added a
     * year from now cannot walk past the accumulator the way it could walk past a hand-picked
     * logging line.
     */
    override suspend fun execute(call: ToolCall): ToolOutcome =
        executeCall(call).also { recordOutcome(it) }

    private val logger = logger()

    /**
     * Position of a call within this round-trip's batch — assigned once, on entry to
     * [executeCall], before any of its four returns. A plain counter (not the ordinal-only-on-
     * success kind): every call that reaches `executeCall` consumes a slot regardless of which
     * exit it takes. [AtomicInteger] because [ParallelToolCallScheduler] may call concurrently
     * (same reason [transitionSignatures] is synchronized).
     */
    private val ordinalCounter = AtomicInteger(0)

    /**
     * Every [ToolOutcome] [execute] has produced so far this round-trip, in the order recorded.
     * Created alongside this whole executor — BEFORE the caller's `try` in
     * [ToolCallingAgent.executeWithToolsInner] — so it survives whatever throws the round-trip
     * out from under [executeTurn]: a broken stream (the `Failed`-event path, where
     * [TurnResult.stats] is still assembled but discarded by the throw) or a tool-job failure
     * (e.g. "Unknown tool: …", which blows up `coroutineScope` before [executeTurn] assembles
     * anything at all). [outcomesSoFar] is the ONLY thing both failure paths have in common — it
     * is what the round-trip's FAILURE record reads its `toolOutcomes` from (the composition of the
     * failed round-trip's record is taken from the accumulator, not from responseMessage). The success
     * path does NOT read this back — it keeps using `toolJobs.awaitAll()`, in the exact order the
     * model issued the calls.
     */
    private val collectedOutcomes = mutableListOf<ToolOutcome>()

    @Synchronized
    private fun recordOutcome(outcome: ToolOutcome) {
        collectedOutcomes += outcome
    }

    /** Snapshot of [collectedOutcomes] — see its KDoc. Order is call-completion order, not
     *  necessarily the model's batch order (a later call may finish before an earlier one under
     *  [ParallelToolCallScheduler]); callers that need the model's order sort by [ToolOutcome.ordinal]. */
    @Synchronized
    fun outcomesSoFar(): List<ToolOutcome> = collectedOutcomes.toList()

    /**
     * [forced error-visibility] Signatures ([ToolCall.transitionSignature]) of the transition gates
     * this round-trip's batch called SUCCESSFULLY — i.e. the phase change(s) the batch actually
     * registered. A [TransitionResult.Failure] (refused gate) is not recorded: nothing is pending,
     * so it is no "place" to hold at. Mutated from tool jobs that a [ParallelToolCallScheduler] may
     * run concurrently, hence the synchronized accessors.
     */
    private val transitionSignatures = mutableListOf<String>()

    @Synchronized
    private fun recordRequestedTransition(signature: String) {
        transitionSignatures += signature
    }

    /** Sorted (order-independent) snapshot of the transitions this batch requested successfully. */
    @Synchronized
    fun requestedTransitions(): List<String> = transitionSignatures.sorted()

    // [mountedInstruments] come from `AgentToolMount.instruments`, which already has
    // `Stateful.defaultUserToolPhases` applied to instruments WITHOUT their own `@Phases` and
    // WITHOUT `@AllPhases` (so `transitionTo`, carrying `@AllPhases`, stays available in every
    // phase). The runtime guard in [executeCall] reads `phases` off this index — provider-level
    // allowlists are not used (incompatible with cached content).
    private val index = mountedInstruments.associateBy { it.name }

    /**
     * Assembles the MODEL+TOOL `PromptContent` blocks from the accumulated events (text parts, tool calls
     * and their results) to add to historyPrompt. The tool results arrive here already
     * computed — the agent `awaitAll()`s the parallel tool jobs before calling this helper.
     */
    fun buildContents(
        parts: List<MessagePart>,
        toolResults: List<ToolResultPromptPart>,
        historyCacheStrategy: PromptCacheStrategy,
    ): List<PromptContent> {
        // Tool history — [historyCacheStrategy] (NO_CACHE unless the model's cache kind is MARKS): every conversation delta (text/tool_call/tool_result)
        // is unique per round-trip, so caching it is useless. Before 2026-05-09 these parts
        // were marked PIPELINE, which did not work under the STRICT cache collection strategy (the parts
        // came after the first NO_CACHE block), and under GROUP_BY_CACHEABILITY it led
        // to cache re-creation through AlwaysGrow on every turn (16 caches per pipeline
        // instead of ~5–6, $0.42 on cache_creation).
        // All model+tool exchanges are marked with a common replaceableSectionId — the snapshot
        // reflects the total effect of all setter/mutator calls, so older exchanges
        // in the conversation become token ballast and are dropped by per-content pruning
        // when a fresh exchange appears on the next round-trip.
        // Parts keep their arrival order; the provider mappers lay calls out last on the wire.
        val modelParts = parts.mapNotNull { part ->
            when (part) {
                is MessagePart.Text -> TextPromptPart(
                    part.text,
                    cacheStrategy = historyCacheStrategy,
                    replaceableSectionId = TOOL_EXCHANGE_HISTORY_SECTION,
                    kind = PartKind.TEXT,
                )
                is MessagePart.Thought -> TextPromptPart(
                    part.text,
                    part.signature,
                    cacheStrategy = historyCacheStrategy,
                    replaceableSectionId = TOOL_EXCHANGE_HISTORY_SECTION,
                    kind = PartKind.THOUGHT,
                )
                is MessagePart.ToolCall -> ToolCallPromptPart(
                    part.toToolCall(),
                    cacheStrategy = historyCacheStrategy,
                    replaceableSectionId = TOOL_EXCHANGE_HISTORY_SECTION,
                )
                else -> null
            }
        }

        return listOf(
            PromptContent(PromptContentType.MODEL, modelParts),
            PromptContent(
                PromptContentType.TOOL,
                toolResults.map {
                    it.copy(
                        cacheStrategy = historyCacheStrategy,
                        replaceableSectionId = TOOL_EXCHANGE_HISTORY_SECTION,
                    )
                }
            )
        )
    }

    suspend fun reportCacheCreated(event: CacheCreated) {
        ec.reportCacheCreate(iterationName, event)
    }

    /**
     * Dispatches [ExecutionContext.onStreamStarted] the moment `executeTurn`'s `collect` sees
     * `StreamResponded` (the provider answered; `PromptStarted` precedes the request) . Fires from inside the stream
     * collection, BEFORE `TurnResult` is assembled: an obryv-stream round-trip throws out of
     * `executeTurn` without ever reaching that assembly, so a dispatch placed after it would
     * silently miss exactly the round-trips this event exists to cover.
     */
    suspend fun reportStreamStarted(modelName: String) {
        ec.onStreamStarted(executionId, modelName)
    }

    private suspend fun executeCall(call: ToolCall): ToolOutcome {
        // Assigned before any of the four returns below, so every call that reaches this
        // function — regardless of which exit it takes — gets a slot in the batch's ordering.
        val ordinal = ordinalCounter.getAndIncrement()

        val instrument = index[call.name] ?: throw MalformedResponseException(
            ParsePhase.TOOL_CALL,
            "Unknown tool: ${call.name}",
            null,
        )

        val stateBefore = stateful.currentState
        val currentPhase = stateBefore.toString()
        if (instrument.phases.isNotEmpty() && currentPhase !in instrument.phases) {
            val message = "Tool '${call.name}' is not available in phase '$currentPhase'. Available phases: ${instrument.phases}"
            logger.warn("Agent {} wrong phase tool call: {}", iterationName, message)
            return ToolOutcome(
                name = call.name,
                args = call.args,
                rendered = JsonPrimitive("Error: $message"),
                isError = true,
                ordinal = ordinal,
                invoked = false,
                exit = ToolOutcome.Exit.PHASE_REFUSED,
                declaredReader = instrument.recordReader,
                thoughtSignature = call.thoughtSignature,
                callId = call.callId,
            )
        }

        // Dynamic substate narrowing on TOP of the static @Phases gate (see
        // [Stateful.availableTools]): a tool otherwise-in-phase may be locked given the builder's
        // live substate (e.g. working mutators once only a `selfCheck` is owed). Enforced here so
        // the lock is real, not merely un-advertised in the phase block.
        val staticallyAllowed = index.values
            .filter { it.phases.isEmpty() || currentPhase in it.phases }
            .map { it.name }
            .toSet()
        val allowedNow = stateful.availableTools(stateBefore, staticallyAllowed)
        if (call.name !in allowedNow) {
            val message = "Tool '${call.name}' is locked in the current substate of phase '$currentPhase'. Callable now: ${allowedNow.joinToString(", ")}."
            logger.warn("Agent {} substate-locked tool call: {}", iterationName, message)
            return ToolOutcome(
                name = call.name,
                args = call.args,
                rendered = JsonPrimitive("Error: $message"),
                isError = true,
                ordinal = ordinal,
                invoked = false,
                exit = ToolOutcome.Exit.LOCKED,
                declaredReader = instrument.recordReader,
                thoughtSignature = call.thoughtSignature,
                callId = call.callId,
            )
        }

        // @NeverSkip instruments opt out of hard-skipping entirely — re-issuing them is always
        // correct (see [NeverSkip]): a control gate's result is a conditional check over mutable
        // state that OTHER calls change between round-trips (skipping would freeze a stale
        // refusal), and a positional mutator's identical args legitimately target a shifted
        // element each time (skipping would silently drop a repeat deletion/update).
        val neverSkip = instrument.annotations.filterIsInstance<NeverSkip>().firstOrNull()

        // Hard no-op skip: this (name,args) already ran ≥2× this invocation with an identical,
        // non-error result, so its effect is proven applied and idempotent — return the stored
        // result (NO-OP-prefixed) WITHOUT re-invoking the tool. The model still "spent" a call, so
        // afterToolCall still fires (call-limit metrics advance and eventually stop a stuck
        // model), but no builder-mutating tool runs and no listeners are told an effect happened.
        // A stored error/refusal is never skipped ([RepeatedCallTracker.skipResultFor]).
        if (neverSkip == null) {
            repeatedCalls.skipResultFor(call.name, call.args)?.let { stored ->
                logger.warn("Agent {} skipping repeated no-op tool call {}({})", iterationName, call.name, call.args)
                ec.afterToolCall(call.name, executionId)
                // skipResultFor only ever returns a NON-error stored result (see its KDoc) —
                // isError is hardcoded false, not recomputed from `stored`.
                return ToolOutcome(
                    name = call.name,
                    args = call.args,
                    rendered = stored.appendNote(REPEATED_CALL_NOOP),
                    isError = false,
                    ordinal = ordinal,
                    invoked = false,
                    exit = ToolOutcome.Exit.REPEAT_SKIPPED,
                    declaredReader = instrument.recordReader,
                    thoughtSignature = call.thoughtSignature,
                    callId = call.callId,
                )
            }
        }

        logger.debug("Agent {} calling tool {}({})", iterationName, call.name, call.args)

        val invocationResult = try {
            // Decoding the arguments, the before-call hook and the invocation share ONE `try`: a
            // refused argument ([ToolArgumentException]) — or a throwing hook — takes the same path
            // as a failing tool (onToolCallError, then an "Error: …" result for the model), the
            // round-trip goes on and the tool is not invoked. onBeforeToolCall is not fired for a
            // call whose arguments were refused: there are no typed arguments to report.
            val args = instrument.prepareWireArguments(call.args)
            ec.dispatchToolCalling { onBeforeToolCall(stateBefore, call.name, args) }
            // The call's name rides in the coroutine context for the duration of this one invocation
            // (see [CurrentToolCall]) — a tool body asks `currentToolName()` instead of being told.
            val result = withContext(CurrentToolCall(call.name)) { instrument.invoke(args) }
            ec.dispatchToolCalling { onAfterToolCall(stateful.currentState, stateBefore, call.name, result) }
            result
        } catch (e: Exception) {
            val actualException = if (e is InvocationTargetException) e.targetException ?: e else e
            // An abort is not a tool result for the model: it leaves the agentic loop as is
            // (checked after unwrapping — a reflective invoke wraps the tool's own exception).
            actualException.rethrowIfAborts()
            if (actualException is ToolArgumentException) {
                // An expected model-side mistake, not a framework failure: no stack trace.
                logger.warn("Agent {} tool {} refused arguments: {}", iterationName, call.name, actualException.message)
            } else {
                logger.error(
                    "Agent $iterationName tool ${call.name} error: ${actualException.message}",
                    actualException
                )
            }
            ec.dispatchToolCalling { onToolCallError(stateful.currentState, call.name, actualException) }
            "Error: ${actualException.message ?: actualException::class.simpleName}"
        }

        // State transitions are applied BETWEEN round-trips (in [Stateful.onAfterToolCalls]), not
        // inside a tool call, so diffing state here would never fire — the transition observability
        // dispatch lives at the apply point in the round-trip loop instead.
        logger.debug("Agent {} tool {} returned: {}", iterationName, call.name, invocationResult)
        ec.afterToolCall(call.name, executionId)

        // A state-transition tool ([Stateful.transitionTo]) returns a [TransitionResult]: the
        // rendered tool result is its human-readable [TransitionResult.message], and the
        // error/refusal classification comes HONESTLY from [TransitionResult.success] — a
        // transition refusal is an error even when its wording ("Close the task, then retry")
        // carries no prefix that [ErrorMatcher] would recognise. Any other result (including a
        // bare-String gate that does NOT return a TransitionResult) falls back to the ErrorMatcher
        // prefix heuristic; an indistinguishable string then counts as SUCCESS (no advisory),
        // never a false rejection.
        val transition = invocationResult as? TransitionResult
        val rendered: Any? = if (transition != null) transition.message else invocationResult
        // [forced error-visibility] A gate that ACCEPTED the transition registered a phase change for
        // the end of this round-trip (or applied it inline, for an immediate applier) — remember it as
        // the "place" a forced error-visibility hold would happen at. Refusals are skipped: they
        // change nothing, and the model sees them on the next round-trip anyway.
        if (transition != null && transition.success) recordRequestedTransition(call.transitionSignature())
        // Record the ORIGINAL result for repeat detection; the first identical repeat of this
        // (name,args,result) gets an advisory appended (subsequent non-error ones are hard-skipped
        // above before ever reaching here) — see [RepeatedCallTracker]. Which advisory depends on
        // what the repeat actually MEANS, not merely on being @NeverSkip:
        // - an error/refusal result (any instrument), including a transition [TransitionResult.Failure]
        //   now classified honestly → the honest [REPEATED_CALL_REFUSAL_ADVISORY] ("rejected,
        //   nothing applied").
        // - a repeat of a @NeverSkip gate/mutator whose result is NOT an error → NO note at all:
        //   the repeat is a legitimate re-application, not a no-op or a rejection, so attaching
        //   either note would misreport a successful call. This covers BOTH use cases:
        //     * a @NeverSkip transition gate that legitimately fires the SAME transition twice in
        //       one run (e.g. WORKING→REVIEW → fix → WORKING→REVIEW again), whose byte-identical
        //       "Successful transition to REVIEW" must NOT be clobbered with a false "REJECTED";
        //     * a @NeverSkip positional mutator (`removePart(index)`) whose constant result hides a
        //       real re-application to a shifted target.
        //   (Because refusals are now caught by [isError] above, both cases are handled uniformly
        //   here — no per-annotation branching needed.)
        // - anything else repeating → the "effect already applied" [REPEATED_CALL_ADVISORY].
        val resultJson = rendered.toJsonElement()
        val isError = if (transition != null) !transition.success else ErrorMatcher.resolveDefault().isError(invocationResult)
        val isRepeat = repeatedCalls.record(call.name, call.args, resultJson, isError)
        val finalResult = when {
            !isRepeat -> resultJson
            isError -> resultJson.appendNote(REPEATED_CALL_REFUSAL_ADVISORY)
            neverSkip != null -> resultJson
            else -> resultJson.appendNote(REPEATED_CALL_ADVISORY)
        }
        return ToolOutcome(
            name = call.name,
            args = call.args,
            rendered = finalResult,
            isError = isError,
            ordinal = ordinal,
            invoked = true,
            exit = ToolOutcome.Exit.INVOKED,
            declaredReader = instrument.recordReader,
            thoughtSignature = call.thoughtSignature,
            callId = call.callId,
        )
    }
}

/** History kind of a response text part: set from the response event type, never inferred from a signature. */
private fun ExecutionTextPart.promptKind(): PartKind = if (isThought) PartKind.THOUGHT else PartKind.TEXT
