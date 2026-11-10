package org.nobirds.murmuration.agent

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import org.nobirds.murmuration.tooling.AgentToolMount
import org.nobirds.murmuration.tooling.Instrument
import org.nobirds.murmuration.prompt.InstrumentDeclaration
import org.nobirds.murmuration.tooling.InstrumentParam
import org.nobirds.murmuration.tooling.TaskCloseResult
import org.nobirds.murmuration.tooling.TaskClosing
import org.nobirds.murmuration.tooling.buildInstruments
import java.io.File
import java.net.URLClassLoader
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

/** Test matcher registered through a service file in a synthetic class loader — never on the real classpath. */
class AlwaysErrorMatcher : ErrorMatcher {
    override fun isError(result: Any?): Boolean = true
}

class NeverErrorMatcher : ErrorMatcher {
    override fun isError(result: Any?): Boolean = false
}

/**
 * Covers the single source of the error protocol (`ErrorMatcher.resolveDefault`): the 0 / 1 / >1
 * implementation resolution rules through the internal class-loader hook, and "ERROR: …" being
 * classified as an error on every classification path that does not go through the agent itself
 * (`buildInstruments`, `AgentToolMount.of`; the agent path is covered by the error-visibility tests).
 *
 * Canonical sample for the hook: build a service file in a temp dir, wrap it in a [URLClassLoader]
 * and call `ErrorMatcherResolver.load(loader)` — the real classpath and the cached resolution stay untouched.
 */
class ErrorMatcherResolutionTest {

    @TempDir
    lateinit var dir: File

    private fun loaderWith(vararg implementations: Class<*>): ClassLoader {
        val services = File(dir, "META-INF/services").apply { mkdirs() }
        File(services, ErrorMatcher::class.java.name).writeText(implementations.joinToString("\n") { it.name })
        return URLClassLoader(arrayOf(dir.toURI().toURL()), ErrorMatcher::class.java.classLoader)
    }

    @Test
    fun `no implementation resolves to the built-in ERROR matcher`() {
        // null parent: the loader sees no service files at all
        val resolved = ErrorMatcherResolver.load(URLClassLoader(emptyArray(), null))
        assertSame(DefaultErrorMatcher, resolved)
        assertTrue(resolved.isError("ERROR: boom"))
        assertTrue(resolved.isError("  error: lower-case, leading whitespace"))
        assertFalse(resolved.isError("all good"))
        assertFalse(resolved.isError(null))
    }

    @Test
    fun `exactly one implementation is taken`() {
        val resolved = ErrorMatcherResolver.load(loaderWith(AlwaysErrorMatcher::class.java))
        assertTrue(resolved is AlwaysErrorMatcher)
    }

    @Test
    fun `more than one implementation fails the resolution`() {
        val ex = assertFailsWith<IllegalStateException> {
            ErrorMatcherResolver.load(loaderWith(AlwaysErrorMatcher::class.java, NeverErrorMatcher::class.java))
        }
        assertTrue(ex.message!!.contains(AlwaysErrorMatcher::class.java.name))
        assertTrue(ex.message!!.contains(NeverErrorMatcher::class.java.name))
    }

    @Test
    fun `public resolution is cached and idempotent`() {
        assertSame(ErrorMatcher.resolveDefault(), ErrorMatcher.resolveDefault())
        assertTrue(ErrorMatcher.resolveDefault().isError("ERROR: x"))
    }

    class Closer : TaskClosing {
        val failed = mutableListOf<String>()
        val closed = mutableListOf<String>()

        override suspend fun closeTaskFromTool(id: String): TaskCloseResult {
            closed += id
            return TaskCloseResult("closed $id", allTasksClosed = false)
        }

        override suspend fun notifyTaskCloseFailed(id: String) {
            failed += id
        }

        @Instrument("Fails with the ERROR protocol.")
        fun failing(@InstrumentParam("Any value") value: String): String = "ERROR: nope"

        @Instrument("Succeeds.")
        fun succeeding(@InstrumentParam("Any value") value: String): String = "fine"
    }

    private suspend fun exercise(closer: Closer, instruments: List<InstrumentDeclaration<*>>) {
        val args = { id: String -> mapOf<String, Any?>("value" to "v", "closesTaskId" to id) }
        instruments.single { it.name == "failing" }.invoke(args("t1"))
        instruments.single { it.name == "succeeding" }.invoke(args("t2"))
        assertEquals(listOf("t1"), closer.failed, "an ERROR-prefixed result must not close the task")
        assertEquals(listOf("t2"), closer.closed)
    }

    @Test
    fun `ERROR result is an error in buildInstruments`() = runBlocking {
        val closer = Closer()
        exercise(closer, buildInstruments { from(closer) })
    }

    @Test
    fun `ERROR result is an error in AgentToolMount of`() = runBlocking {
        val closer = Closer()
        exercise(closer, AgentToolMount.of(closer, AgentToolMount.statefulOf(closer)).instruments)
    }
}
