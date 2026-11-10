package org.nobirds.murmuration.openai

import com.openai.errors.BadRequestException
import com.openai.errors.InternalServerException
import com.openai.errors.NotFoundException
import com.openai.errors.OpenAIIoException
import com.openai.errors.OpenAIServiceException
import com.openai.errors.PermissionDeniedException
import com.openai.errors.RateLimitException
import com.openai.errors.SseException
import com.openai.errors.UnauthorizedException
import com.openai.errors.UnprocessableEntityException
import com.openai.models.responses.Response
import com.openai.models.responses.ResponseStatus
import kotlinx.serialization.SerializationException
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
import java.io.IOException
import java.io.InterruptedIOException
import kotlin.jvm.optionals.getOrNull
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

/**
 * Maps exceptions from the Azure OpenAI SDK / low-level I/O into provider-neutral
 * [PromptExecutionException]s, at the executor boundary — mirror of `GeminiExceptionMapper`.
 * Every exception that escapes `:murmuration-openai` is already typed, so consumers
 * (`RetryStrategy`, `RecoveryAgent`, statistics, DB) work with types instead of parsing
 * `e.message`.
 *
 * If the throwable is already a [PromptExecutionException] it is returned as-is (so the
 * executor may throw the mapped types directly and still funnel `catch` input through here).
 */
fun Throwable.toPromptExecutionException(prompt: Prompt): PromptExecutionException {
    if (this is PromptExecutionException) return this
    return when (this) {
        // 429: retryable quota/rate-limit. Retry-After (if the header is present) drives the
        // per-attempt backoff.
        is RateLimitException -> QuotaExceededException(
            kind = quotaKindFromMessage(message),
            retryAfter = retryAfterFrom(this),
            message = message ?: "Rate limit (429)",
            cause = this,
        )
        // 400: Azure rejects policy-violating INPUT with a 400 carrying a content-filter marker
        // — that's a policy violation, not a malformed request; everything else 400 is a real
        // invalid request (unfixable by retry, caller must rewrite the prompt).
        is BadRequestException -> classifyBadRequest(prompt, this)
        // 401: the Entra token may have expired on the request boundary; retry pulls a fresh
        // one from the credential supplier.
        is UnauthorizedException ->
            TransientException(TransientReason.AUTH_REFRESH, message ?: "Unauthorized (401)", this)
        // 403: a config/permission problem — retrying is pointless.
        is PermissionDeniedException ->
            InvalidRequestException(prompt, httpCode = 403, message = message ?: "Permission denied (403)", this)
        // 404: typically a wrong Azure deployment name.
        is NotFoundException ->
            InvalidRequestException(
                prompt,
                httpCode = 404,
                message = message ?: "Not found (404) — check the Azure deployment name",
                cause = this,
            )
        is UnprocessableEntityException ->
            InvalidRequestException(prompt, httpCode = 422, message = message ?: "Unprocessable entity (422)", this)
        is InternalServerException ->
            TransientException(TransientReason.SERVER_ERROR, message ?: "Server error (5xx)", this)
        is SseException ->
            TransientException(TransientReason.NETWORK_ERROR, message ?: "SSE stream error", this)
        is OpenAIIoException ->
            TransientException(TransientReason.NETWORK_ERROR, message ?: "Network/I/O error", this)
        is InterruptedIOException ->
            TransientException(TransientReason.NETWORK_TIMEOUT, message ?: "Network timeout", this)
        is IOException -> mapIoException(this)
        is SerializationException ->
            MalformedResponseException(ParsePhase.SERIALIZATION, message ?: "Serialization error", this)
        else -> mapByMessage(this) ?: UnknownProviderException(
            this::class.qualifiedName ?: this::class.simpleName ?: "Unknown",
            message ?: "Unknown provider error",
            this,
        )
    }
}

/**
 * A short telemetry token describing a [Response]'s terminal status — the Responses API analogue
 * of chat completions' `finish_reason`. Flows into [PromptExecutionEvent.Completed.finishReason]
 * and [org.nobirds.murmuration.prompt.StreamDiagnostics.lastChunkFinishReason]. `completed` covers
 * both a plain stop and a tool-call turn (a response carrying function calls still completes
 * normally).
 */
internal fun Response.finishToken(): String = when (status().getOrNull()) {
    null, ResponseStatus.COMPLETED -> "completed"
    ResponseStatus.INCOMPLETE ->
        incompleteDetails().getOrNull()?.reason()?.getOrNull()?.toString()?.lowercase() ?: "incomplete"
    ResponseStatus.FAILED -> "failed"
    ResponseStatus.IN_PROGRESS -> "in_progress"
    else -> status().getOrNull()?.toString()?.lowercase() ?: "completed"
}

/**
 * Maps a [Response]'s terminal status into a provider-neutral exception, or `null` for a normal
 * completion. The Responses API replaces chat's `finish_reason` with a `status` +
 * `incomplete_details.reason` pair: `incomplete/max_output_tokens` → [MaxTokensException],
 * `incomplete/content_filter` (or a content-filter `failed`) → [PolicyViolationException].
 */
internal fun Response.mapOutcome(prompt: Prompt): PromptExecutionException? = mapResponseOutcome(
    status = status().getOrNull(),
    incompleteReason = incompleteDetails().getOrNull()?.reason()?.getOrNull()?.toString(),
    errorMessage = error().getOrNull()?.message(),
    prompt = prompt,
)

/**
 * Pure status → exception mapping, split out from [mapOutcome] so it can be unit-tested without
 * building a full [Response]. `null` status is treated as completed (a well-formed response always
 * carries one).
 */
internal fun mapResponseOutcome(
    status: ResponseStatus?,
    incompleteReason: String?,
    errorMessage: String?,
    prompt: Prompt,
): PromptExecutionException? = when (status) {
    null, ResponseStatus.COMPLETED -> null
    ResponseStatus.INCOMPLETE -> when (incompleteReason?.lowercase()) {
        "max_output_tokens" -> MaxTokensException(prompt, "Response truncated: max_output_tokens", null)
        "content_filter" ->
            PolicyViolationException(prompt, PolicyCategory.PROHIBITED, "content_filter", "Blocked by content filter", null)
        else -> TransientException(
            TransientReason.INCOMPLETE_RESPONSE,
            "Response incomplete: ${incompleteReason ?: "unknown reason"}",
            cause = null,
        )
    }
    ResponseStatus.FAILED -> {
        val msg = errorMessage ?: "Response failed"
        if (isContentFilter(msg))
            PolicyViolationException(prompt, PolicyCategory.PROHIBITED, "content_filter", msg, null)
        else
            UnknownProviderException("ResponseStatus.failed", msg, null)
    }
    ResponseStatus.IN_PROGRESS -> TransientException(
        TransientReason.INCOMPLETE_RESPONSE,
        "Response still in progress when the stream ended",
        cause = null,
    )
    else -> UnknownProviderException("ResponseStatus", "Unknown response status: $status", null)
}

/**
 * Reads the `Retry-After` header (whole seconds) from a service exception, when present.
 * A message-regex fallback is intentionally omitted — the header carries the value on Azure's
 * 429 responses and is sufficient.
 */
private fun retryAfterFrom(e: OpenAIServiceException): Duration? =
    e.headers().values("Retry-After").firstOrNull()?.toLongOrNull()?.seconds

/**
 * Typed error codes ([BadRequestException.code]) that always mean a content-filter rejection,
 * checked before falling back to substring matching on [message]. Lower-cased for comparison.
 * `content_filter`/`contentfilter` — Azure text/chat completions; `responsibleaipolicyviolation`
 * — the Azure RAI inner-error code surfaced as the top-level code on some API versions.
 * `moderation_blocked` is intentionally excluded here — it gets its own [PolicyCategory.SAFETY]
 * branch in [classifyBadRequest] because it's the gpt-image safety classifier, not the text
 * content-policy filter.
 */
private val CONTENT_FILTER_CODES = setOf("content_filter", "contentfilter", "responsibleaipolicyviolation")

/** `safety_violations=[sexual]` / `safety_violations=[sexual, violence]` from a gpt-image
 * moderation-blocked message — captures the bracket contents verbatim as the violation list. */
private val SAFETY_VIOLATIONS_REGEX = Regex("""safety_violations=\[([^]]*)]""")

/**
 * Classifies a [BadRequestException] into [PolicyViolationException] (content/safety rejection)
 * or [InvalidRequestException] (real malformed request). Structured data first — [code] when the
 * SDK exposes it, then a `safety_violations=[...]` marker in [message] — falling back to
 * substring matching on the raw message for API shapes without a typed code.
 *
 * gpt-image safety-classifier rejections (`code = "moderation_blocked"`, message containing
 * `safety_violations=[...]`) map to [PolicyCategory.SAFETY] — mirroring Gemini's
 * `SAFETY`/`IMAGE_SAFETY` — with the violated categories captured verbatim as [finishReason].
 * Azure's text/chat content-policy filter (`content_filter` family) keeps mapping to
 * [PolicyCategory.PROHIBITED], as before.
 */
private fun classifyBadRequest(prompt: Prompt, e: BadRequestException): PromptExecutionException {
    val msg = e.message ?: "Bad request (400)"
    val code = e.code().getOrNull()?.lowercase()
    val safetyViolations = SAFETY_VIOLATIONS_REGEX.find(msg)?.groupValues?.get(1)?.trim()?.takeIf { it.isNotEmpty() }
    return when {
        code == "moderation_blocked" || safetyViolations != null ->
            PolicyViolationException(prompt, PolicyCategory.SAFETY, safetyViolations ?: code ?: "moderation_blocked", msg, e)
        code in CONTENT_FILTER_CODES || isContentFilter(msg) ->
            PolicyViolationException(prompt, PolicyCategory.PROHIBITED, code ?: "content_filter", msg, e)
        else -> InvalidRequestException(prompt, httpCode = 400, message = msg, cause = e)
    }
}

/**
 * Substring fallback for content-filter/safety markers when [BadRequestException.code] isn't a
 * typed match (or the caller — e.g. [mapResponseOutcome]'s FAILED branch — has no typed code at
 * all, only a message).
 */
private fun isContentFilter(message: String): Boolean =
    listOf(
        "content_filter",
        "content management policy",
        "responsibleaipolicyviolation",
        "rejected by the safety system",
        "safety_violations=",
    ).any { it in message.lowercase() }

private fun mapIoException(e: IOException): PromptExecutionException {
    val msg = e.message?.lowercase() ?: ""
    val reason = if ("timeout" in msg) TransientReason.NETWORK_TIMEOUT else TransientReason.NETWORK_ERROR
    return TransientException(reason, e.message ?: "I/O error", e)
}

/**
 * Substring fallback for exceptions that bypassed the typed SDK classes (e.g. wrapped in a
 * plain `RuntimeException` from a side-channel call). Returns `null` when nothing matches, so
 * the caller emits [UnknownProviderException].
 */
private fun mapByMessage(e: Throwable): PromptExecutionException? {
    val msg = e.message?.lowercase() ?: return null
    return when {
        "too many requests" in msg || "rate limit" in msg || "429" in msg ->
            QuotaExceededException(quotaKindFromMessage(msg), retryAfter = null, message = e.message ?: "Quota", cause = e)
        "timeout" in msg || "deadline" in msg ->
            TransientException(TransientReason.NETWORK_TIMEOUT, e.message ?: "Timeout", e)
        "overloaded" in msg || "service unavailable" in msg || "internal server error" in msg ->
            TransientException(TransientReason.SERVER_ERROR, e.message ?: "Server error", e)
        "connection reset" in msg || "connection" in msg ->
            TransientException(TransientReason.NETWORK_ERROR, e.message ?: "Connection error", e)
        else -> null
    }
}

private fun quotaKindFromMessage(msg: String?): QuotaKind {
    val lower = msg?.lowercase() ?: ""
    return when {
        "resource exhausted" in lower -> QuotaKind.RESOURCE_EXHAUSTED
        "rate limit" in lower || "too many" in lower -> QuotaKind.RATE_LIMIT
        else -> QuotaKind.RATE_LIMIT
    }
}
