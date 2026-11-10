package org.nobirds.murmuration.gemini

import com.google.genai.errors.ClientException
import com.google.genai.errors.GenAiIOException
import com.google.genai.errors.ServerException
import kotlinx.serialization.SerializationException
import org.nobirds.murmuration.prompt.CacheExpiredException
import org.nobirds.murmuration.prompt.InvalidRequestException
import org.nobirds.murmuration.prompt.MalformedResponseException
import org.nobirds.murmuration.prompt.MaxTokensException
import org.nobirds.murmuration.prompt.ParsePhase
import org.nobirds.murmuration.prompt.PolicyCategory
import org.nobirds.murmuration.prompt.PolicyViolationException
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionException
import org.nobirds.murmuration.prompt.QuotaExceededException
import org.nobirds.murmuration.prompt.QuotaKind
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.UnknownProviderException
import org.nobirds.murmuration.retry.findServerRetryAfterSeconds
import java.io.IOException
import java.io.InterruptedIOException
import kotlin.time.Duration.Companion.seconds

/**
 * Maps exceptions from the Gemini SDK / low-level I/O into provider-neutral
 * [PromptExecutionException]s. Called on the executor boundary: every exception that
 * escapes `:murmuration-gemini` is already typed, so consumers (`RetryStrategy`,
 * `RecoveryAgent`, statistics, DB) work with types instead of parsing `e.message`.
 *
 * Takes [prompt] for the prompt-related branches ([PolicyViolationException],
 * [MaxTokensException], [InvalidRequestException]) — these types carry the prompt so
 * recovery logic can rewrite it without side-channel plumbing. The executor always has the
 * prompt in scope at the call site, so the parameter is not nullable.
 *
 * If the throwable is already a [PromptExecutionException] it's returned as-is. That lets
 * the executor throw the new types directly (e.g. `TransientException` for an empty
 * stream) and still safely funnel any `catch` input through this mapper.
 */
fun Throwable.toPromptExecutionException(prompt: Prompt): PromptExecutionException {
    if (this is PromptExecutionException) return this
    return when (this) {
        is ClientException -> mapClientException(this, prompt)
        is ServerException -> mapServerException(this)
        // `GenAiIOException` extends `RuntimeException` (not `java.io.IOException`), so it
        // wouldn't be caught by the `is IOException` branch below. Wraps SSE-stream / SDK
        // parse failures — message is typically "Failed to parse the JSON string." when a
        // chunk arrives truncated. Always retryable: a fresh request usually completes.
        is GenAiIOException -> TransientException(
            TransientReason.NETWORK_ERROR,
            message ?: "GenAI I/O error (stream chunk truncated or unparseable)",
            this,
        )
        is InterruptedIOException -> TransientException(
            TransientReason.NETWORK_TIMEOUT,
            message ?: "Network timeout",
            this,
        )
        is IOException -> mapIoException(this)
        is SerializationException -> MalformedResponseException(
            ParsePhase.SERIALIZATION,
            message ?: "Serialization error",
            this,
        )
        else -> mapByMessage(this, prompt) ?: UnknownProviderException(
            this::class.qualifiedName ?: this::class.simpleName ?: "Unknown",
            message ?: "Unknown provider error",
            this,
        )
    }
}

/**
 * Maps a Gemini `FinishReason` (as the string returned by `candidate.finishReason()`) into
 * a provider-neutral exception. Returns `null` for `STOP` (normal completion) — the caller
 * then has no error to throw and decides what to do (e.g. the image branch treats
 * "STOP without content" as a separate [MalformedResponseException]).
 */
fun mapFinishReason(finishReason: String?, prompt: Prompt): PromptExecutionException? {
    val token = finishReason
        // Stream produced chunks but server never emitted a terminating finishReason.
        // Observed with Gemini flash-preview after long thinking phases (single thought
        // chunk arrives, then SSE closes without STOP/MAX_TOKENS/SAFETY/etc.). Retry-safe:
        // a fresh request typically completes normally, so route through TransientException
        // instead of UnknownProviderException (which DefaultRetryStrategy does not retry).
        ?: return TransientException(
            TransientReason.INCOMPLETE_RESPONSE,
            "Provider ended stream without a finish reason",
            cause = null,
        )
    return when (token.uppercase()) {
        "STOP" -> null
        "MAX_TOKENS" -> MaxTokensException(prompt, "Response truncated: MAX_TOKENS", null)
        "SAFETY", "IMAGE_SAFETY" ->
            PolicyViolationException(prompt, PolicyCategory.SAFETY, token, "Blocked by safety filter ($token)", null)
        "PROHIBITED_CONTENT", "IMAGE_PROHIBITED_CONTENT", "BLOCKLIST" ->
            PolicyViolationException(prompt, PolicyCategory.PROHIBITED, token, "Blocked by content policy ($token)", null)
        "RECITATION", "IMAGE_RECITATION" ->
            PolicyViolationException(prompt, PolicyCategory.RECITATION, token, "Blocked by recitation filter ($token)", null)
        "SPII" ->
            PolicyViolationException(prompt, PolicyCategory.SPII, token, "Blocked by SPII filter", null)
        "OTHER", "IMAGE_OTHER" ->
            PolicyViolationException(prompt, PolicyCategory.OTHER, token, "Blocked: $token", null)
        "MALFORMED_FUNCTION_CALL" ->
            MalformedResponseException(ParsePhase.FUNCTION_CALL, "Model returned malformed function call", null)
        "UNEXPECTED_TOOL_CALL" ->
            MalformedResponseException(ParsePhase.TOOL_CALL, "Model returned unexpected tool call", null)
        "LANGUAGE" ->
            InvalidRequestException(prompt, httpCode = null, message = "Unsupported language", cause = null)
        "FINISH_REASON_UNSPECIFIED" ->
            UnknownProviderException("FinishReason.UNSPECIFIED", "Provider didn't specify finish reason", null)
        else ->
            UnknownProviderException("FinishReason.$token", "Unknown finish reason: $token", null)
    }
}

private fun mapClientException(e: ClientException, prompt: Prompt): PromptExecutionException {
    val code = e.code()
    val msg = e.message ?: "ClientException $code"
    return when {
        // 499 "operation was cancelled" — the request was aborted, typically because an upstream
        // deadline (an external round-trip ceiling / dropped client) elapsed before the model
        // replied. This is NOT a bad request: a fresh attempt usually completes, so route it
        // through the retryable [TransientException] (mirrors the NETWORK_TIMEOUT branch for
        // [InterruptedIOException]) instead of the non-retryable [InvalidRequestException] the
        // generic client-error `else` below would produce — otherwise a single external timeout
        // kills the whole pipeline with no retry.
        code == 499 || msg.contains("operation was cancelled", ignoreCase = true) ->
            TransientException(TransientReason.NETWORK_TIMEOUT, msg, e)
        code == 429 -> e.toQuotaExceeded()
        code == 400 -> when {
            msg.contains("Cache content", ignoreCase = true) && msg.contains("expired", ignoreCase = true) ->
                CacheExpiredException(cacheId = null, message = msg, cause = e)
            msg.contains("mint used too late", ignoreCase = true) || msg.contains("gaiamint", ignoreCase = true) ->
                TransientException(TransientReason.AUTH_REFRESH, msg, e)
            else ->
                InvalidRequestException(prompt, httpCode = 400, message = msg, cause = e)
        }
        else -> InvalidRequestException(prompt, httpCode = code, message = msg, cause = e)
    }
}

/**
 * The one place a raw genai 429 becomes a [QuotaExceededException]. Used by [toPromptExecutionException]
 * and directly by callers that have no [Prompt] (cache create/refresh), so the quota ladder of
 * `RetrySession` sees the same type on every path.
 */
internal fun ClientException.toQuotaExceeded(): QuotaExceededException {
    val msg = message ?: "ClientException ${code()}"
    return quotaExceeded(msg, this)
}

/** The one constructor of [QuotaExceededException] here; the retry-after hint comes from the shared [findServerRetryAfterSeconds]. */
private fun quotaExceeded(msg: String, cause: Throwable): QuotaExceededException = QuotaExceededException(
    kind = quotaKindFromMessage(msg),
    retryAfter = cause.findServerRetryAfterSeconds()?.seconds,
    message = msg,
    cause = cause,
)

/** [toQuotaExceeded] for a 429 [ClientException] when the throwable is one (or already a [QuotaExceededException]); `null` for anything else. */
internal fun Throwable.asQuotaExceededOrNull(): QuotaExceededException? = when {
    this is QuotaExceededException -> this
    this is ClientException && code() == 429 -> toQuotaExceeded()
    else -> null
}

private fun mapServerException(e: ServerException): PromptExecutionException =
    TransientException(
        TransientReason.SERVER_ERROR,
        e.message ?: "Server error ${e.code()}",
        e,
    )

private fun mapIoException(e: IOException): PromptExecutionException {
    val msg = e.message?.lowercase() ?: ""
    val reason = when {
        "timeout" in msg -> TransientReason.NETWORK_TIMEOUT
        else -> TransientReason.NETWORK_ERROR
    }
    return TransientException(reason, e.message ?: "I/O error", e)
}

/**
 * Substring-based fallback for exceptions that didn't arrive via the typed SDK classes.
 * Covers leftover cases: cache-expired/mint wrapped in RuntimeException, "too many
 * requests" in text, etc. If nothing matches, returns `null` and the caller emits
 * [UnknownProviderException].
 */
private fun mapByMessage(e: Throwable, prompt: Prompt): PromptExecutionException? {
    val msg = e.message?.lowercase() ?: return null
    return when {
        "cache content" in msg && "expired" in msg ->
            CacheExpiredException(cacheId = null, message = e.message ?: "Cache expired", cause = e)
        "mint used too late" in msg || "gaiamint invalid" in msg ->
            TransientException(TransientReason.AUTH_REFRESH, e.message ?: "Auth refresh", e)
        "too many requests" in msg || "resource exhausted" in msg || "429" in msg ->
            quotaExceeded(e.message ?: "Quota", e)
        "deadline" in msg || "timeout" in msg || "operation was cancelled" in msg ->
            TransientException(TransientReason.NETWORK_TIMEOUT, e.message ?: "Timeout", e)
        "service unavailable" in msg || "internal server error" in msg ->
            TransientException(TransientReason.SERVER_ERROR, e.message ?: "Server error", e)
        else -> null
    }
}

private fun quotaKindFromMessage(msg: String): QuotaKind {
    val lower = msg.lowercase()
    return when {
        "resource exhausted" in lower -> QuotaKind.RESOURCE_EXHAUSTED
        "rate limit" in lower || "too many" in lower -> QuotaKind.RATE_LIMIT
        else -> QuotaKind.UNSPECIFIED
    }
}
