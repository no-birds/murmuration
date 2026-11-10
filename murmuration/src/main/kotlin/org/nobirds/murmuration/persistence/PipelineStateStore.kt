package org.nobirds.murmuration.persistence

import org.nobirds.murmuration.agent.StepStatus
import java.util.UUID
import kotlin.time.Instant

/**
 * Lifecycle status of a pipeline run, as seen by the durable state store.
 *
 * [RUNNING] is written the moment the pipeline starts (before any round-trip completes), so a
 * crash or a client reconnect can always find an in-flight pipeline. [SUCCESS]/[FAILED] are
 * terminal.
 */
enum class PipelineRunStatus { RUNNING, SUCCESS, FAILED }

/**
 * Durable snapshot of a single pipeline run — the lifecycle "header". One per `pipelineId`.
 *
 * Intentionally carries only **identity, scope and lifecycle** — not progress. Progress is derived
 * elsewhere: `completed` is a count over per-round-trip statistics, the estimate is a UX heuristic,
 * and "which agents are running now" is multi-valued (agents run in parallel) and lives in the
 * round-trip statistics (RUNNING rows), not a header field. Keeping those out of here is what makes
 * the header race-free under parallel agents (only `start`/`finish` ever touch it).
 *
 * Scope fields ([gameId]/[turnId]/[characterId]/[userId]/[bookId]) mirror the request scope and are
 * all nullable — a given pipeline type only fills the ones it has.
 */
data class PipelineRunState(
    val pipelineId: UUID,
    val pipelineName: String,
    val status: PipelineRunStatus,
    val userId: UUID?,
    val gameId: UUID?,
    val turnId: UUID?,
    val characterId: UUID?,
    val bookId: UUID?,
    val errorMessage: String?,
    val startedAt: Instant,
    val updatedAt: Instant,
    val finishedAt: Instant?,
    /**
     * Correlation key of the logical *operation* this run belongs to. Two runs separated by a
     * durable handoff (the second one starts in a different process, possibly after a restart)
     * cannot share a record — a live run does not survive a restart — but they can share a root,
     * which is the only honest way to answer "what did this one player request cost".
     *
     * It is an id of an operation, not of a run: whoever owns the operation mints it (typically
     * a fresh synthetic id) *before* the first fragment exists, so it need not match any row of
     * the runs table at all. A run that is nobody's fragment is its own operation and carries its
     * own [pipelineId] here — which is also how the pre-root history is backfilled — but that is
     * a convenient default, not the definition.
     */
    val rootId: UUID = pipelineId,
    /** Optional label of the operation the root groups (e.g. the dispatcher's entry point). */
    val operationKind: String? = null,
)

/**
 * SPI for persisting and reading pipeline run state. Defined in the framework so persistence is a
 * framework feature: the host application supplies the implementation (a DB-backed one here, a
 * file-backed one for a standalone runtime). Consumers (player snapshot, admin views) depend on
 * this interface, not on the concrete storage — when pipeline execution is split into its own
 * service, only the read implementation changes (to a REST client), consumers stay put.
 *
 * The header has only two writes — [start] and [finish] — and both fire once per pipeline (before
 * any agent / after all agents), so there is no per-agent state here and nothing for parallel
 * agents to race on. [stepStarted] / [stepFinished] write a *different* record (the composition
 * nodes of the run), keyed by their own id, so they don't race the header either.
 * Implementations must never throw: a storage hiccup must not break a running pipeline.
 */
interface PipelineStateStore {

    /**
     * Inserts the RUNNING header at pipeline start.
     *
     * [rootId] is the correlation key of the logical operation (see [PipelineRunState.rootId]) —
     * defaults to the run's own id, which is what a standalone run wants; a dispatcher that
     * drives several runs for one request passes its own root. [operationKind] labels that
     * operation and is free-form.
     */
    suspend fun start(
        pipelineId: UUID,
        pipelineName: String,
        userId: UUID?,
        gameId: UUID?,
        turnId: UUID?,
        characterId: UUID?,
        bookId: UUID?,
        rootId: UUID = pipelineId,
        operationKind: String? = null,
    )

    /** Moves the header to a terminal [status], stamping `finishedAt` and an optional error. */
    suspend fun finish(pipelineId: UUID, status: PipelineRunStatus, errorMessage: String?)

    /**
     * Records an opened composition node ([org.nobirds.murmuration.agent.step] frame) of a run.
     * [stepId] is the node's own key — [path] is not unique on its own (the same module may run
     * twice under one parent). Default no-op: a store that only cares about the header stays
     * valid, and the framework keeps working when nothing listens.
     *
     * The node's own name is deliberately *not* a parameter: it is the last segment of [path],
     * and a store that shortens the path (column widths are the store's business) is the only
     * place that knows which segment survived. Passing it separately guaranteed exactly one bug —
     * a name that no longer occurs in the path it was supposedly taken from.
     */
    suspend fun stepStarted(
        pipelineId: UUID,
        stepId: UUID,
        path: List<String>,
        startedAt: Instant,
    ) {}

    /**
     * Closes the node opened by [stepStarted] with its terminal [status] and, when it failed,
     * the error that left the frame. This is the only durable trace of a step that never
     * performed an LLM round-trip.
     */
    suspend fun stepFinished(
        pipelineId: UUID,
        stepId: UUID,
        status: StepStatus,
        finishedAt: Instant,
        errorMessage: String?,
    ) {}

    /**
     * Safety net fired once, when the run itself ends: whatever frames of [pipelineId] are still
     * RUNNING never got their finish report and never will, so they are closed as failed.
     *
     * A frame is a *pair* of writes (RUNNING on open, terminal on close), and the second one can
     * be lost for reasons the frame cannot defend against by itself — the storage call failed,
     * the process died between them. Without this sweep such a node stays RUNNING forever and
     * reads as "still in flight" long after the pipeline is over. Same shape as the orphan sweep
     * over `llm_statistics` RUNNING rows, for the same reason.
     */
    suspend fun stepsAbandoned(pipelineId: UUID, finishedAt: Instant) {}

    /** Active (RUNNING) runs scoped to a game — drives player progress restore. */
    suspend fun activeByGame(gameId: UUID): List<PipelineRunState>

    /** All active (RUNNING) runs — drives the admin live view. */
    suspend fun allActive(): List<PipelineRunState>

    /**
     * No-op store. Used as a default when no persistence is wired (tests, standalone runs that
     * don't care about durable state). Reads return empty.
     */
    object NoOp : PipelineStateStore {
        override suspend fun start(
            pipelineId: UUID,
            pipelineName: String,
            userId: UUID?,
            gameId: UUID?,
            turnId: UUID?,
            characterId: UUID?,
            bookId: UUID?,
            rootId: UUID,
            operationKind: String?,
        ) {}

        override suspend fun finish(pipelineId: UUID, status: PipelineRunStatus, errorMessage: String?) {}
        override suspend fun activeByGame(gameId: UUID): List<PipelineRunState> = emptyList()
        override suspend fun allActive(): List<PipelineRunState> = emptyList()
    }
}
