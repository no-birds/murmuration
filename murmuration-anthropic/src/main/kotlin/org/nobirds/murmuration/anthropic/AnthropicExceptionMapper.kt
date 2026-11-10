package org.nobirds.murmuration.anthropic

import com.anthropic.errors.AnthropicIoException
import com.anthropic.errors.AnthropicServiceException
import com.anthropic.errors.BadRequestException
import com.anthropic.errors.InternalServerException
import com.anthropic.errors.NotFoundException
import com.anthropic.errors.PermissionDeniedException
import com.anthropic.errors.RateLimitException
import com.anthropic.errors.SseException
import com.anthropic.errors.UnauthorizedException
import com.anthropic.errors.UnprocessableEntityException
import com.anthropic.models.ErrorType
import kotlinx.serialization.SerializationException
import org.nobirds.murmuration.prompt.InvalidRequestException
import org.nobirds.murmuration.prompt.MalformedResponseException
import org.nobirds.murmuration.prompt.ParsePhase
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
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds

/** HTTP status Anthropic uses for "overloaded" (the SDK has no dedicated class for it). */
internal const val ANTHROPIC_OVERLOADED_STATUS = 529

/**
 * Hint attached to the 400 that the API returns when an identity-linked key is used without
 * `anthropic-workspace-id`. Configuration problem, not a prompt problem: retrying cannot fix it.
 */
internal const val WORKSPACE_ID_HINT =
    "Anthropic rejected the request because the workspace id is missing: set the workspace id " +
        "for an identity-linked key (ANTHROPIC_WORKSPACE_ID / ai.anthropic.workspaceId)."

/**
 * Maps exceptions from the Anthropic SDK / low-level I/O into provider-neutral
 * [PromptExecutionException]s at the executor boundary — mirror of `OpenAiExceptionMapper`.
 * Messages are built from the status and the API's error text only; request headers (and with them
 * the API key) are never printed.
 *
 * The SDK is configured with `maxRetries(0)`, so every retry decision is made by our layer on the
 * mapped types ([AnthropicRetryStrategy] + `DefaultRetryStrategy`).
 *
 * A [PromptExecutionException] is returned as-is.
 */
fun Throwable.toPromptExecutionException(prompt: Prompt): PromptExecutionException {
    if (this is PromptExecutionException) return this
    return when (this) {
        is RateLimitException -> QuotaExceededException(
            kind = QuotaKind.RATE_LIMIT,
            retryAfter = retryAfterFrom(this),
            message = serviceMessage(this, "Rate limit (429)"),
            cause = this,
        )
        is BadRequestException -> classifyBadRequest(prompt, this)
        // Authentication / permission: a key or federation configuration problem, retry is pointless.
        is UnauthorizedException -> InvalidRequestException(
            prompt, httpCode = 401, message = serviceMessage(this, "Unauthorized (401) — check the Anthropic credentials"), cause = this,
        )
        is PermissionDeniedException -> InvalidRequestException(
            prompt, httpCode = 403, message = serviceMessage(this, "Permission denied (403)"), cause = this,
        )
        is NotFoundException -> InvalidRequestException(
            prompt, httpCode = 404, message = serviceMessage(this, "Not found (404) — check the model name"), cause = this,
        )
        is UnprocessableEntityException -> InvalidRequestException(
            prompt, httpCode = 422, message = serviceMessage(this, "Unprocessable entity (422)"), cause = this,
        )
        // 5xx incl. 529 overloaded — transient.
        is InternalServerException ->
            TransientException(TransientReason.SERVER_ERROR, serviceMessage(this, "Server error (${statusCode()})"), this)
        // An `overloaded_error` event inside a stream arrives as SseException.
        is SseException -> mapSse(this)
        is AnthropicServiceException -> mapOtherStatus(prompt, this)
        is AnthropicIoException ->
            TransientException(TransientReason.NETWORK_ERROR, message ?: "Network/I/O error", this)
        is InterruptedIOException ->
            TransientException(TransientReason.NETWORK_TIMEOUT, message ?: "Network timeout", this)
        is IOException -> mapIoException(this)
        is SerializationException ->
            MalformedResponseException(ParsePhase.SERIALIZATION, message ?: "Serialization error", this)
        else -> UnknownProviderException(
            this::class.qualifiedName ?: this::class.simpleName ?: "Unknown",
            message ?: "Unknown provider error",
            this,
        )
    }
}

/**
 * Delay from `Retry-After-Ms` (preferred, as in the SDK's own retry logic) or `Retry-After`
 * (whole seconds; the HTTP-date form is not parsed). Only [QuotaExceededException] carries it —
 * the core's [TransientException] has no delay field.
 */
internal fun retryAfterFrom(e: AnthropicServiceException): Duration? =
    e.headers().values("Retry-After-Ms").firstOrNull()?.trim()?.toDoubleOrNull()?.takeIf { it >= 0 }?.milliseconds
        ?: e.headers().values("Retry-After").firstOrNull()?.trim()?.toLongOrNull()?.takeIf { it >= 0 }?.seconds

/** `true` for HTTP 529 or an `overloaded_error` body (also the mid-stream error event). */
internal fun AnthropicServiceException.isOverloaded(): Boolean =
    statusCode() == ANTHROPIC_OVERLOADED_STATUS || errorType().getOrNull() == ErrorType.OVERLOADED_ERROR

private fun mapSse(e: SseException): PromptExecutionException =
    if (e.isOverloaded()) {
        TransientException(TransientReason.SERVER_ERROR, serviceMessage(e, "Anthropic overloaded (stream error event)"), e)
    } else {
        TransientException(TransientReason.NETWORK_ERROR, serviceMessage(e, "SSE stream error"), e)
    }

private fun classifyBadRequest(prompt: Prompt, e: BadRequestException): PromptExecutionException {
    val text = serviceMessage(e, "Bad request (400)")
    val message = if (mentionsWorkspaceId(e, text)) "$WORKSPACE_ID_HINT API said: $text" else text
    return InvalidRequestException(prompt, httpCode = 400, message = message, cause = e)
}

private fun mentionsWorkspaceId(e: AnthropicServiceException, text: String): Boolean =
    "anthropic-workspace-id" in text.lowercase() || "anthropic-workspace-id" in e.body().toString().lowercase()

/** Statuses without a dedicated SDK class: 408/409 are transient (the SDK's own policy), other 4xx are fatal. */
private fun mapOtherStatus(prompt: Prompt, e: AnthropicServiceException): PromptExecutionException {
    val code = e.statusCode()
    return when {
        code == 408 -> TransientException(TransientReason.NETWORK_TIMEOUT, serviceMessage(e, "Request timeout (408)"), e)
        code == 409 -> TransientException(TransientReason.SERVER_ERROR, serviceMessage(e, "Conflict (409)"), e)
        code == 429 -> QuotaExceededException(QuotaKind.RATE_LIMIT, retryAfterFrom(e), serviceMessage(e, "Rate limit (429)"), e)
        code >= 500 -> TransientException(TransientReason.SERVER_ERROR, serviceMessage(e, "Server error ($code)"), e)
        code in 400..499 -> InvalidRequestException(prompt, httpCode = code, message = serviceMessage(e, "Client error ($code)"), cause = e)
        else -> UnknownProviderException(e::class.qualifiedName ?: "AnthropicServiceException", serviceMessage(e, "Unexpected status ($code)"), e)
    }
}

private fun serviceMessage(e: AnthropicServiceException, fallback: String): String =
    e.message?.takeIf { it.isNotBlank() } ?: fallback

private fun mapIoException(e: IOException): PromptExecutionException {
    val reason = if ("timeout" in (e.message?.lowercase() ?: "")) TransientReason.NETWORK_TIMEOUT else TransientReason.NETWORK_ERROR
    return TransientException(reason, e.message ?: "I/O error", e)
}
