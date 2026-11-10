package org.nobirds.murmuration.openai

import com.openai.errors.InternalServerException
import com.openai.errors.OpenAIIoException
import com.openai.errors.RateLimitException
import com.openai.errors.SseException
import org.nobirds.murmuration.retry.RetryStrategy
import org.nobirds.murmuration.retry.causes

/**
 * Azure OpenAI-specific retry triggers: the raw SDK exception types (`429`, `5xx`, SSE and
 * network I/O errors) plus a substring fallback for messages that arrive without a structural
 * exception. Mirror of `GeminiRetryStrategy`.
 *
 * The mapped provider-neutral [org.nobirds.murmuration.prompt.PromptExecutionException] types
 * are already covered by
 * [DefaultRetryStrategy][org.nobirds.murmuration.retry.DefaultRetryStrategy]; this strategy
 * exists to catch the raw SDK exceptions that bypass the mapper (e.g. from side-channel
 * calls). Compose the two with `DefaultRetryStrategy or OpenAiRetryStrategy`.
 */
object OpenAiRetryStrategy : RetryStrategy {
    private val MESSAGES = listOf(
        "rate limit", "too many requests", "overloaded", "timeout", "connection reset",
    )

    override fun shouldRetry(throwable: Throwable, attempt: Int): Boolean =
        throwable.causes.any { it.matches() }

    private fun Throwable.matches(): Boolean {
        val msg = message?.lowercase() ?: ""
        return when (this) {
            is RateLimitException -> true
            is InternalServerException -> true
            is SseException -> true
            is OpenAIIoException -> true
            else -> MESSAGES.any { it in msg }
        }
    }
}
