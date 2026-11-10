package org.nobirds.murmuration.agent

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.retry.RetryStrategy
import java.util.UUID
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.time.Duration

/**
 * Contract of the composition frames introduced by [step]: what the path is built from, which
 * invocations open a node, and what reaches a listener when a node fails. These are the
 * guarantees the durable side (`pipeline_steps`, `llm_statistics.step_path`) is derived from —
 * if a frame stops nesting, or a wrapper starts showing up as its own node, the stored tree
 * silently stops matching the code.
 */
class StepFramesTest {

    private data class FinishedStep(
        val stepId: UUID,
        val path: List<String>,
        val status: StepStatus,
        val error: String?,
    )

    private class StepRecorder : PipelineEventListener {
        val started = ConcurrentLinkedQueue<Pair<UUID, List<String>>>()
        val finished = ConcurrentLinkedQueue<FinishedStep>()
        val entries = ConcurrentLinkedQueue<AgentExecutionEntry>()

        override suspend fun onStepStarted(pipelineId: UUID, stepId: UUID, path: List<String>) {
            started += stepId to path
        }

        override suspend fun onStepFinished(
            pipelineId: UUID,
            stepId: UUID,
            path: List<String>,
            status: StepStatus,
            duration: Duration,
            error: String?,
        ) {
            finished += FinishedStep(stepId, path, status, error)
        }

        override suspend fun onEntryReported(pipelineId: UUID, entry: AgentExecutionEntry) {
            entries += entry
        }

        fun paths(): List<List<String>> = finished.map { it.path }
    }

    @Test
    fun `nested steps build the path and stamp it on entries reported inside`(): Unit = runBlocking {
        val recorder = StepRecorder()

        executePipeline(pipelineEventListener = recorder) {
            step("SceneCycle") {
                step("Staging") {
                    reportAgentExecution(LambdaAgentExecutionEntry("Cartographer", Duration.ZERO, true))
                }
            }
        }

        assertEquals(
            listOf(listOf("SceneCycle", "Staging"), listOf("SceneCycle")),
            recorder.paths(),
            "inner frame must close first, and each frame must carry its full path",
        )
        assertEquals(
            listOf("SceneCycle", "Staging"),
            recorder.entries.single().stepPath,
            "an entry must be stamped with the path of the frame it was reported from",
        )
    }

    @Test
    fun `entries reported at the pipeline root stay flat`(): Unit = runBlocking {
        val recorder = StepRecorder()

        executePipeline(pipelineEventListener = recorder) {
            reportAgentExecution(LambdaAgentExecutionEntry("RootWork", Duration.ZERO, true))
        }

        assertEquals(
            emptyList(),
            recorder.entries.single().stepPath,
            "no frame — no path: this is the legacy shape every stored record must keep having",
        )
        assertTrue(recorder.finished.isEmpty(), "no frame was opened, so no step may be reported")
    }

    @Test
    fun `executing an agent opens a frame named after the agent, and nesting follows invocation`(): Unit = runBlocking {
        val recorder = StepRecorder()

        val worker = agent<Unit, String>("Worker") { "sheet" }
        val order = agent<Unit, String>("SheetOrder") { execute(worker, Unit) }

        executePipeline(pipelineEventListener = recorder) {
            step("Reply") { execute(order, Unit) }
        }

        assertEquals(
            listOf(
                listOf("Reply", "SheetOrder", "Worker"),
                listOf("Reply", "SheetOrder"),
                listOf("Reply"),
            ),
            recorder.paths(),
            "the tree of agent invocations must be the tree of steps",
        )
        // The very point of the path: the same module under a different parent is a different
        // node, while the code stays one.
        val workerEntry = recorder.entries.first { it.agentName.startsWith("Worker") }
        assertEquals(listOf("Reply", "SheetOrder", "Worker"), workerEntry.stepPath)
    }

    @Test
    fun `the same module under two parents produces two distinct paths`(): Unit = runBlocking {
        val recorder = StepRecorder()
        val order = agent<Unit, String>("SheetOrder") { "sheet" }

        executePipeline(pipelineEventListener = recorder) {
            step("Reply") { execute(order, Unit) }
            step("Staging") { execute(order, Unit) }
        }

        assertTrue(listOf("Reply", "SheetOrder") in recorder.paths())
        assertTrue(listOf("Staging", "SheetOrder") in recorder.paths())
    }

    @Test
    fun `step name strips the generated instance id, the invocation counter and the type suffix`() {
        assertEquals("TurnPlannerAgent", "TurnPlannerAgent#0e66#2[PLANNING][CallToolAgent]".toStepName())
        assertEquals("SheetOrder", "SheetOrder#a1b2[Lambda]".toStepName())
        assertEquals("PlainName", "PlainName".toStepName())
        // A path is stored and queried as its segments joined by '/', so a name carrying one would
        // forge a node that never ran and break every prefix query below it.
        assertEquals("Order_Resolver", "Order/Resolver".toStepName())
    }

    @Test
    fun `a retry wrapper is transparent - retried attempts stay inside one frame`(): Unit = runBlocking {
        val recorder = StepRecorder()
        var attempts = 0

        val flaky = agent<Unit, String>("FlakyAgent") {
            attempts++
            if (attempts < 3) throw IllegalStateException("boom $attempts")
            "ok"
        }.withRetry(
            maxRetries = 5,
            initialBackoff = Duration.ZERO,
            retryStrategy = RetryStrategy { _, _ -> true },
        )

        executePipeline(pipelineEventListener = recorder) { execute(flaky, Unit) }

        assertEquals(3, attempts, "sanity: the delegate really was retried")
        assertEquals(
            listOf(listOf("FlakyAgent")),
            recorder.paths(),
            "a retry is a continuation of the same logical node — not a new frame, and not a 'Retry' segment",
        )
    }

    @Test
    fun `a step that throws is reported FAILED with the error and rethrows`(): Unit = runBlocking {
        val recorder = StepRecorder()

        // A deterministic sub-step that dies in code, not on a round-trip: it produces no LLM
        // statistics at all, so the frame is its only trace.
        val broken = agent<Unit, String>("OrderResolver") { error("no such character") }

        assertFailsWith<IllegalStateException> {
            executePipeline(pipelineEventListener = recorder) {
                step("Reply") { execute(broken, Unit) }
            }
        }

        val inner = recorder.finished.first { it.path == listOf("Reply", "OrderResolver") }
        assertEquals(StepStatus.FAILED, inner.status)
        assertEquals("no such character", inner.error)

        val outer = recorder.finished.first { it.path == listOf("Reply") }
        assertEquals(StepStatus.FAILED, outer.status, "the failure must propagate through the enclosing frame too")

        assertTrue(
            recorder.started.any { it.first == inner.stepId },
            "start and finish of one frame must be correlatable by step id",
        )
    }

    @Test
    fun `a frame cancelled by a failing sibling is still closed`(): Unit = runBlocking {
        val recorder = StepRecorder()
        val siblingEntered = CompletableDeferred<Unit>()

        // The failure mode this guards: a store writes RUNNING when a frame opens and moves it to a
        // terminal state only on the finish report. Report inside a cancelled coroutine and the
        // report never happens — the node stays RUNNING forever and reads as "still in flight" long
        // after the pipeline is over, i.e. a cancellation is indistinguishable from a crashed pod.
        assertFailsWith<IllegalStateException> {
            executePipeline(pipelineEventListener = recorder) {
                coroutineScope {
                    async { step("Sibling") { siblingEntered.complete(Unit); awaitCancellation() } }
                    async { siblingEntered.await(); step("Boom") { error("boom") } }
                }
            }
        }

        val sibling = recorder.finished.single { it.path == listOf("Sibling") }
        assertEquals(StepStatus.FAILED, sibling.status, "a cancelled frame must still reach a terminal state")
        assertTrue(
            recorder.started.any { it.first == sibling.stepId },
            "the closed frame must be the one that was opened, not a synthesised row",
        )
        assertEquals(StepStatus.FAILED, recorder.finished.single { it.path == listOf("Boom") }.status)
    }

    @Test
    fun `the failing frame keeps the message capped, ancestors keep only a pointer to it`(): Unit = runBlocking {
        val recorder = StepRecorder()
        val huge = "x".repeat(2000)
        val broken = agent<Unit, String>("OrderResolver") { error(huge) }

        assertFailsWith<IllegalStateException> {
            executePipeline(pipelineEventListener = recorder) {
                step("Reply") { step("SheetOrder") { execute(broken, Unit) } }
            }
        }

        val origin = recorder.finished.single { it.path.last() == "OrderResolver" }
        assertEquals(
            STEP_ERROR_MAX_LENGTH + STEP_ERROR_TRUNCATION_MARKER.length,
            origin.error!!.length,
            "the frame that failed keeps the message — capped, so one runaway text cannot flood the store",
        )
        assertTrue(origin.error!!.endsWith(STEP_ERROR_TRUNCATION_MARKER), "a cut message must never read as a complete one")

        // The same exception travels through every enclosing frame. Storing its full text on each
        // of them is what used to write five copies of one failure and lose where it started.
        listOf("SheetOrder", "Reply").forEach { ancestor ->
            val error = recorder.finished.single { it.path.last() == ancestor }.error!!
            assertTrue(error.startsWith("failed in OrderResolver: "), "an ancestor must point at the origin, got: $error")
            assertTrue(error.length < origin.error!!.length, "an ancestor must not carry a second full copy")
        }
    }

    @Test
    fun `coroutines started inside a frame inherit its path`(): Unit = runBlocking {
        val recorder = StepRecorder()
        val leaf = agent<Int, Int>("Leaf") { it }

        executePipeline(pipelineEventListener = recorder) {
            step("Fanout") {
                (1..3).map { i -> async { execute(leaf, i) } }.awaitAll()
            }
        }

        val leafSteps = recorder.finished.filter { it.path == listOf("Fanout", "Leaf") }
        assertEquals(3, leafSteps.size, "each parallel child must open its own node under the same path")
        assertEquals(3, leafSteps.map { it.stepId }.toSet().size, "sibling nodes must be distinguishable by id")
        assertNotNull(recorder.finished.firstOrNull { it.path == listOf("Fanout") })
    }
}
