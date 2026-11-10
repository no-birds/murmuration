package org.nobirds.murmuration.agent

import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withContext
import java.util.Collections
import java.util.UUID
import java.util.WeakHashMap
import kotlin.time.TimeSource

/**
 * Terminal outcome of a step frame (see [step]). [RUNNING] is the state a durable store writes
 * the moment a frame opens, so an in-flight (or crashed) step is visible before it finishes;
 * [SUCCESS] / [FAILED] are terminal.
 */
enum class StepStatus { RUNNING, SUCCESS, FAILED }

/**
 * A composition node inside a single pipeline run.
 *
 * The pipeline record stays flat by design (one process invocation = one `pipelineId`); a step
 * frame is what gives that record an internal *structure* without splitting it: the same module
 * executed under two different parents shows up as two different paths
 * (`Reply/SheetOrder/Worker` vs `Staging/SheetOrder/Worker`) while the code, the
 * warm cache and the call budget stay shared.
 *
 * Propagation is lexical, through the receiver: [block] runs with a child [ExecutionContext]
 * whose [ExecutionContext.stepPath] is the parent path plus [name]. Everything executed with
 * that receiver — nested [step] calls, `execute(agent, input)`, coroutines started on it —
 * inherits the path automatically; anything handed the *parent* context deliberately does not.
 *
 * Every round-trip reported through the child context is stamped with the path
 * ([AgentExecutionEntry.stepPath]), and the frame itself is reported to
 * [PipelineEventListener.onStepStarted] / [PipelineEventListener.onStepFinished] — including
 * steps that never touch an LLM, which is the only way a failed deterministic sub-step reaches
 * durable storage at all.
 *
 * Frames are transparent to the budget and to progress counting: they do not start a new
 * `PipelineRun`, do not reset the call limit, and do not emit agent START/END events.
 *
 * **Closing a frame outlives cancellation.** The finish report runs under [NonCancellable]: a
 * store writes RUNNING when the frame opens and only the finish report moves it to a terminal
 * state, so losing that report to a cancelled parent (a sibling failed, the caller timed out)
 * would leave the node RUNNING *forever* — the failure would look like a crash instead of a
 * cancellation. Cancellation itself is not swallowed: the throwable is rethrown right after the
 * frame is closed.
 */
suspend fun <T> ExecutionContext.step(name: String, block: suspend ExecutionContext.() -> T): T {
    val child = StepExecutionContext(this, name)
    val stepId = UUID.randomUUID()
    reportStepStarted(stepId, child.stepPath)
    val startMark = TimeSource.Monotonic.markNow()
    val result = try {
        with(child) { block() }
    } catch (e: Throwable) {
        try {
            withContext(NonCancellable) {
                reportStepFinished(
                    stepId,
                    child.stepPath,
                    StepStatus.FAILED,
                    startMark.elapsedNow(),
                    e.stepErrorMessage(name),
                )
            }
        } catch (reportFailure: Throwable) {
            // The original failure is what the caller must see; a broken telemetry sink may not
            // replace it. Keep the report failure attached so a silently dying store is still
            // diagnosable from the very stack trace it corrupted.
            e.addSuppressed(reportFailure)
        }
        throw e
    }
    withContext(NonCancellable) {
        reportStepFinished(stepId, child.stepPath, StepStatus.SUCCESS, startMark.elapsedNow(), null)
    }
    return result
}

/**
 * Child context of a step frame. Delegates the whole [ExecutionContext] contract to [parent]
 * (same pipeline id, same listeners, same budget, same coroutine scope) and adds exactly two
 * things: its own [stepPath], and the stamping of that path onto every entry reported through it.
 *
 * Note it delegates rather than wraps the event stream: `reportStepStarted` / `reportStepFinished`
 * travel up to the [PipelineRun], so listeners see one flat stream of frames whose nesting is
 * carried by the path itself, not by the call stack.
 */
private class StepExecutionContext(
    private val parent: ExecutionContext,
    stepName: String,
) : ExecutionContext by parent {

    override val stepPath: List<String> = parent.stepPath + stepName

    /**
     * Stamps this frame's path onto entries that don't have one yet. An entry that already
     * carries a path was reported by a deeper frame and merely passes through — overwriting it
     * would flatten the tree back to the outermost frame.
     */
    override suspend fun reportAgentExecution(entry: AgentExecutionEntry) {
        parent.reportAgentExecution(
            if (entry.stepPath.isEmpty()) entry.withStepPath(stepPath) else entry
        )
    }
}

/** Longest error text stored on the frame where the failure actually happened. */
internal const val STEP_ERROR_MAX_LENGTH = 300

/** Longest excerpt an *ancestor* frame keeps of a failure that happened below it. */
private const val STEP_ERROR_ANCESTOR_EXCERPT = 120

/** Appended to a message that had to be cut, so a truncated text never reads as a complete one. */
internal const val STEP_ERROR_TRUNCATION_MARKER = "… (truncated)"

/**
 * The frame that first reported a given throwable — i.e. the one where it actually originated.
 * Weak-keyed and identity-based (`Throwable` does not override `equals`), so an entry dies with
 * the exception it describes and nothing has to be evicted by hand.
 *
 * This is what tells an origin frame from the ancestors the same exception then travels through:
 * frames close innermost-first, so the *first* claim on a throwable is by definition the deepest
 * one. Without it, one failure deep in the tree wrote its full text onto every ancestor row —
 * five copies of the same stack-ish message, and no way to tell where it started.
 */
private val originFrameOf: MutableMap<Throwable, String> =
    Collections.synchronizedMap(WeakHashMap<Throwable, String>())

/**
 * Error text stored for a frame closing on [this] throwable, seen from the frame named
 * [frameName]:
 *  - the frame where the failure originated keeps the message itself, capped;
 *  - every enclosing frame keeps only a pointer — `failed in <origin>: <excerpt>` — because the
 *    full text is already stored once, at the place that knows what it means.
 */
private fun Throwable.stepErrorMessage(frameName: String): String {
    val own = message?.takeIf { it.isNotBlank() } ?: this::class.simpleName ?: this::class.toString()
    val origin = originFrameOf.putIfAbsent(this, frameName)
    return if (origin == null) own.capped(STEP_ERROR_MAX_LENGTH)
    else "failed in $origin: ${own.capped(STEP_ERROR_ANCESTOR_EXCERPT)}"
}

private fun String.capped(limit: Int): String =
    if (length <= limit) this else take(limit) + STEP_ERROR_TRUNCATION_MARKER

/**
 * Step name for an agent invocation: the agent's plain name, with the generated instance id,
 * the invocation counter and the `[Type]` suffix stripped
 * (`TurnPlannerAgent#0e66#2[PLANNING][CallToolAgent]` → `TurnPlannerAgent`).
 *
 * The decorated forms exist to make a single *round-trip* row identifiable; a step path is the
 * opposite question — "which module ran under which parent" — and must stay stable across
 * invocations, otherwise every re-run of the same agent would land in its own bucket and the
 * path could never be aggregated by prefix. Repeated invocations under one parent therefore
 * share a path and are told apart by their rows' timestamps.
 *
 * `/` is replaced: a path is stored and queried as its segments joined by that character, so an
 * agent whose name contains one would forge a node that never ran and break every prefix query
 * below it. Agent names come from call sites all over the codebase — sanitizing here is cheaper
 * than trusting each of them.
 */
internal fun String.toStepName(): String {
    val withoutType = substringBefore('[')
    val withoutInstanceId = withoutType.substringBefore('#')
    return withoutInstanceId.ifEmpty { this }.replace('/', '_')
}
