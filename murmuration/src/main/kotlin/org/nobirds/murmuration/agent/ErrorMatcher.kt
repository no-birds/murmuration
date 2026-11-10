package org.nobirds.murmuration.agent

import java.util.ServiceLoader

/**
 * SPI for classifying a tool-call result as "error" vs "ok". The framework uses it in every
 * place that has to decide whether a tool result is an error: the forced error-visibility
 * round-trip of `ToolCallingAgent`, the `closesTaskId` handling of `TaskClosingDecorator`, and
 * the instruments built by `buildInstruments` / `AgentToolMount.of`.
 *
 * There is exactly ONE source of the matcher: [resolveDefault]. It is not configurable per
 * agent, per decorator or per builder, so two code paths can never disagree about what an
 * error is.
 *
 * Resolution goes through [ServiceLoader]:
 *  - no implementation registered: the built-in matcher is used — a case-insensitive prefix
 *    match on `ERROR`;
 *  - exactly one implementation registered: it is used;
 *  - more than one implementation registered: [IllegalStateException] on the first resolution.
 *
 * Contract: the `META-INF/services/org.nobirds.murmuration.agent.ErrorMatcher` file is provided
 * ONLY by the final application. Libraries built on top of murmuration must never register an
 * implementation — two libraries (or a library and the application) would collide and make
 * resolution fail. The result is cached for the lifetime of the JVM, so tests must not swap the
 * matcher through a service file either.
 *
 * Resolution is lazy and happens at the first use. An application that wants a misconfiguration
 * to fail at startup rather than in the middle of a turn calls [resolveDefault] once during
 * its initialization.
 */
fun interface ErrorMatcher {
    fun isError(result: Any?): Boolean

    companion object {
        /**
         * Returns the process-wide matcher (see the type KDoc for the resolution rules). The
         * result is cached; subsequent calls are cheap and return the same instance.
         *
         * @throws IllegalStateException if more than one implementation is registered.
         */
        fun resolveDefault(): ErrorMatcher = ErrorMatcherResolver.cached
    }
}

/**
 * Built-in matcher: treats a result as an error when its string representation starts with
 * `ERROR` (case-insensitive, leading whitespace ignored).
 */
internal object DefaultErrorMatcher : ErrorMatcher {
    private const val PREFIX = "ERROR"

    override fun isError(result: Any?): Boolean {
        val text = result?.toString().orEmpty().trimStart()
        return text.startsWith(PREFIX, ignoreCase = true)
    }
}

/** Resolution of the process-wide [ErrorMatcher]; the only place that talks to [ServiceLoader]. */
internal object ErrorMatcherResolver {
    /** Cached result of [load] on the default class loader. A failed resolution is not cached and throws again. */
    val cached: ErrorMatcher by lazy { load(ErrorMatcher::class.java.classLoader) }

    /**
     * Uncached resolution through [loader] — test hook, the same rules as [ErrorMatcher.resolveDefault].
     */
    fun load(loader: ClassLoader?): ErrorMatcher {
        val found = ServiceLoader.load(ErrorMatcher::class.java, loader).toList()
        return when (found.size) {
            0 -> DefaultErrorMatcher
            1 -> found.single()
            else -> throw IllegalStateException(
                "More than one ErrorMatcher implementation is registered via ServiceLoader: " +
                    found.joinToString { it::class.java.name } +
                    ". Only the final application may register one.",
            )
        }
    }
}
