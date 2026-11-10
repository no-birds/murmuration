package org.nobirds.murmuration.retry

import org.nobirds.murmuration.prompt.CacheExpiredException
import org.nobirds.murmuration.prompt.MalformedResponseException
import org.nobirds.murmuration.prompt.MaxTokensException
import org.nobirds.murmuration.prompt.PolicyViolationException
import org.nobirds.murmuration.prompt.QuotaExceededException
import org.nobirds.murmuration.prompt.TransientException

import kotlinx.serialization.SerializationException
import java.io.IOException
import java.io.InterruptedIOException

/**
 * SPI for the retry decision over a thrown exception. Composing strategies via [or] lets the
 * core ship [DefaultRetryStrategy] (typed `PromptExecutionException` subclasses plus a few
 * structural fallbacks for raw exceptions that bypassed any provider mapper), while provider
 * modules add their own triggers (for example `GeminiRetryStrategy` in `:murmuration-gemini`).
 *
 * Used by [RetryAgent] and [retryWithTimeout]. If `shouldRetry` returns `false`, the exception
 * is propagated without further attempts.
 */
fun interface RetryStrategy {
    fun shouldRetry(throwable: Throwable, attempt: Int): Boolean

    /** Compose: retry if either strategy says so. */
    infix fun or(other: RetryStrategy): RetryStrategy =
        RetryStrategy { t, a -> shouldRetry(t, a) || other.shouldRetry(t, a) }
}

/**
 * Default core strategy. Provider/SDK-agnostic: covers the provider-neutral
 * [PromptExecutionException][org.nobirds.murmuration.prompt.PromptExecutionException]
 * subclasses produced by per-provider exception mappers
 * ([TransientException], [QuotaExceededException], [CacheExpiredException]) plus a couple of
 * structural fallbacks for raw exceptions thrown outside the provider boundary —
 * [InterruptedIOException] (timeouts/cancels), [SerializationException] (JSON parsing), and
 * [IOException] with timeout/connection/reset/canceled in the message.
 *
 * Provider-specific SDK exceptions are not expected to reach this strategy directly — they
 * should be mapped on the boundary of the provider module (e.g. `GeminiExceptionMapper` in
 * `:murmuration-gemini`). For paths that bypass the mapper (cache backend, side-channel API
 * calls), compose a per-provider strategy via [or].
 */
object DefaultRetryStrategy : RetryStrategy {
    private val IO_MESSAGES = listOf("timeout", "connection", "reset", "canceled")

    /**
     * How many times a [PolicyViolationException] (safety / content-filter block) is retried
     * before it becomes an honest fatal. See the branch in [matches] for the rationale.
     */
    const val POLICY_VIOLATION_MAX_RETRIES: Int = 2

    override fun shouldRetry(throwable: Throwable, attempt: Int): Boolean =
        throwable.causes.any { it.matches(attempt) }

    private fun Throwable.matches(attempt: Int): Boolean {
        val msg = message?.lowercase() ?: ""
        return when (this) {
            is TransientException -> true
            is QuotaExceededException -> true
            is CacheExpiredException -> true
            // Safety / content-filter block (SAFETY, PROHIBITED, RECITATION, …). The class
            // docstring calls this deterministic, but real Azure/Gemini traffic proves the
            // content filter is *stochastic* — the very same fixture passes on one attempt and
            // trips on the next. Killing the whole pipeline on a single unlucky roll (the
            // 2026-07 incident) is therefore wrong: give the block a small, bounded number of
            // retries and only then fail honestly. Provider-neutral on purpose — both mappers
            // emit this exact type, so one gate here covers Gemini and Azure alike instead of a
            // copy in each module. The cap is expressed via [attempt] (1-based failed-attempt
            // counter from RetrySession); note it only bites when the caller threads the real
            // attempt (an application-defined strategy) — the attempt-discarding framework
            // default still bounds it by the session's maxRetries.
            is PolicyViolationException -> attempt <= POLICY_VIOLATION_MAX_RETRIES
            // MAX_TOKENS is non-deterministic with thinking enabled: a retry on the same prompt
            // often takes a different trajectory and fits into the cap (especially when the model
            // drowned in a thinking rabbit hole on the first attempt). The cost of an attempt is small
            // (a capped RT is counted in statistics with 0 output tokens — we do not pay for output),
            // 3 retries via RetryAgent are a balance between catching a stochastic variant and
            // protecting against a real loop (when the model has truly looped —
            // it will exit on the 3rd attempt and the pipeline will fail honestly).
            is MaxTokensException -> true
            // MALFORMED_FUNCTION_CALL / SERIALIZATION / UNEXPECTED_TOOL_CALL — the model
            // generated a structurally broken response, but this is also stochastic: each attempt
            // independently samples the function-call JSON, a retry often gets a correct
            // variant. The class docstring already admitted "retrying sometimes works" — now it is
            // aligned with the behaviour. The same 3 attempts of RetryAgent.
            is MalformedResponseException -> true
            is InterruptedIOException -> true
            is SerializationException -> true
            is IOException if IO_MESSAGES.any { it in msg } -> true
            else -> false
        }
    }
}
