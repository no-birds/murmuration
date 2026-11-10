package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.prompt.CallIds
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.ToolCall
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ToolCallSchedulerTest {

    private fun call(name: String) = ToolCall(name, emptyMap(), callId = CallIds.mint())
    private fun result(name: String, value: String): ToolOutcome = ToolOutcome(
        name = name,
        args = emptyMap(),
        rendered = JsonPrimitive(value),
        isError = false,
        ordinal = 0,
        invoked = true,
        exit = ToolOutcome.Exit.INVOKED,
        declaredReader = null,
        callId = CallIds.mint(),
    )

    /** Test executor — plays the role of ToolsExecutor.executeCall. */
    private fun executor(action: suspend (ToolCall) -> ToolOutcome): ToolExecutor =
        ToolExecutor { action(it) }

    @Test
    fun `sequential executes calls in submission order`() = runBlocking {
        coroutineScope {
            val order = mutableListOf<String>()
            val exec = executor { c ->
                order += "start-${c.name}"
                delay(20)
                order += "end-${c.name}"
                result(c.name, c.name)
            }
            SequentialToolCallScheduler().open(this, exec).use { session ->
                val jobs = listOf("a", "b", "c").map { id -> async { session.execute(call(id)) } }
                val results = jobs.awaitAll()
                assertEquals(
                    listOf("start-a", "end-a", "start-b", "end-b", "start-c", "end-c"),
                    order,
                )
                assertEquals(listOf("a", "b", "c"), results.map { it.name })
            }
        }
    }

    @Test
    fun `sequential calls do not overlap in time`() = runBlocking {
        coroutineScope {
            var inFlight = 0
            var maxConcurrent = 0
            val exec = executor { c ->
                inFlight++
                maxConcurrent = maxOf(maxConcurrent, inFlight)
                delay(15)
                inFlight--
                result(c.name, "ok")
            }
            SequentialToolCallScheduler().open(this, exec).use { session ->
                val jobs = (1..5).map { i -> async { session.execute(call("c$i")) } }
                jobs.awaitAll()
                assertEquals(1, maxConcurrent, "Sequential keeps at most one task running")
            }
        }
    }

    @Test
    fun `parallel calls overlap in time`() = runBlocking {
        coroutineScope {
            var inFlight = 0
            var maxConcurrent = 0
            val exec = executor { c ->
                inFlight++
                maxConcurrent = maxOf(maxConcurrent, inFlight)
                delay(30)
                inFlight--
                result(c.name, "ok")
            }
            ParallelToolCallScheduler().open(this, exec).use { session ->
                val jobs = (1..5).map { i -> async { session.execute(call("p$i")) } }
                jobs.awaitAll()
                assertTrue(maxConcurrent >= 2, "Parallel must run calls concurrently, max=$maxConcurrent")
            }
        }
    }

    @Test
    fun `sequential failed call propagates to submitter but session keeps working`() = runBlocking {
        coroutineScope {
            val exec = executor { c ->
                if (c.name == "fails") error("boom")
                result(c.name, "ok")
            }
            SequentialToolCallScheduler().open(this, exec).use { session ->
                assertFailsWith<IllegalStateException> { session.execute(call("fails")) }
                val after = session.execute(call("after"))
                assertEquals("ok", (after.rendered as JsonPrimitive).content)
            }
        }
    }

    @Test
    fun `sequential sessions are independent`() = runBlocking {
        coroutineScope {
            val scheduler = SequentialToolCallScheduler()
            var inFlight = 0
            var maxConcurrent = 0
            val exec = executor { c ->
                inFlight++; maxConcurrent = maxOf(maxConcurrent, inFlight)
                delay(40); inFlight--
                result(c.name, "ok")
            }
            val sessionA = scheduler.open(this, exec)
            val sessionB = scheduler.open(this, exec)
            val jobA = async { sessionA.execute(call("a")) }
            val jobB = async { sessionB.execute(call("b")) }
            jobA.await(); jobB.await()
            sessionA.close(); sessionB.close()
            assertEquals(2, maxConcurrent, "Different sessions are independent, max=$maxConcurrent")
        }
    }

    @Test
    fun `sequential execute suspends caller until exec completes`() = runBlocking {
        coroutineScope {
            val gate = CompletableDeferred<Unit>()
            val exec = executor { c ->
                gate.await()
                result(c.name, "done")
            }
            SequentialToolCallScheduler().open(this, exec).use { session ->
                val res = async { session.execute(call("slow")) }
                gate.complete(Unit)
                assertEquals("done", (res.await().rendered as JsonPrimitive).content)
            }
        }
    }
}
