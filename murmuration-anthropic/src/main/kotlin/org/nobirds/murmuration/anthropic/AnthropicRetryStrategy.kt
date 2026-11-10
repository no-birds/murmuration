package org.nobirds.murmuration.anthropic

import com.anthropic.errors.AnthropicIoException
import com.anthropic.errors.AnthropicServiceException
import com.anthropic.errors.InternalServerException
import com.anthropic.errors.RateLimitException
import com.anthropic.errors.SseException
import org.nobirds.murmuration.retry.RetryStrategy
import org.nobirds.murmuration.retry.causes

/**
 * Anthropic-specific retry triggers for raw SDK exceptions that bypass the mapper (side-channel
 * calls): `429`, `5xx` (incl. 529 overloaded), `408`/`409`, SSE stream errors and network I/O.
 * Mirror of `OpenAiRetryStrategy`. Mapped [org.nobirds.murmuration.prompt.PromptExecutionException]s
 * are covered by [org.nobirds.murmuration.retry.DefaultRetryStrategy]; compose with
 * `DefaultRetryStrategy or AnthropicRetryStrategy`.
 *
 * Fatal statuses (400/401/403/404/422) are never retried — in particular a 400 about the missing
 * workspace id. The SDK's own retries are off (`maxRetries(0)`), so this is the only retry layer.
 */
object AnthropicRetryStrategy : RetryStrategy {
    private val MESSAGES = listOf("rate limit", "too many requests", "overloaded", "timeout", "connection reset")
    private val RETRYABLE_STATUSES = setOf(408, 409, 429)

    override fun shouldRetry(throwable: Throwable, attempt: Int): Boolean =
        throwable.causes.any { it.matches() }

    private fun Throwable.matches(): Boolean = when (this) {
        is RateLimitException, is InternalServerException, is SseException, is AnthropicIoException -> true
        is AnthropicServiceException -> statusCode() in RETRYABLE_STATUSES || statusCode() >= 500
        else -> MESSAGES.any { it in (message?.lowercase() ?: "") }
    }
}
