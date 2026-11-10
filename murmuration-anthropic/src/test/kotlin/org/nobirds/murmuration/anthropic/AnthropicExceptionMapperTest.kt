package org.nobirds.murmuration.anthropic

import com.anthropic.core.JsonValue
import com.anthropic.core.http.Headers
import com.anthropic.errors.AnthropicIoException
import com.anthropic.errors.AnthropicServiceException
import com.anthropic.errors.BadRequestException
import com.anthropic.errors.InternalServerException
import com.anthropic.errors.NotFoundException
import com.anthropic.errors.PermissionDeniedException
import com.anthropic.errors.RateLimitException
import com.anthropic.errors.SseException
import com.anthropic.errors.UnauthorizedException
import com.anthropic.errors.UnexpectedStatusCodeException
import com.anthropic.errors.UnprocessableEntityException
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.InvalidRequestException
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionException
import org.nobirds.murmuration.prompt.QuotaExceededException
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.UnknownProviderException
import java.io.IOException
import java.io.InterruptedIOException
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds

/**
 * Unit tests for [toPromptExecutionException] and [AnthropicRetryStrategy]. SDK exceptions are
 * built with their own builders (real types, not mocks), shaped like the API's error body.
 */
class AnthropicExceptionMapperTest {

    private val prompt = Prompt(AnthropicLlms.findModel("anthropic", "claude-sonnet-5-5"), null, emptyList())

    private fun errorBody(type: String, message: String): JsonValue =
        JsonValue.from(mapOf("type" to "error", "error" to mapOf("type" to type, "message" to message)))

    private fun headers(vararg pairs: Pair<String, String>): Headers =
        Headers.builder().apply { pairs.forEach { (k, v) -> put(k, v) } }.build()

    private fun rateLimit(vararg h: Pair<String, String>) = RateLimitException.builder()
        .headers(headers(*h)).body(errorBody("rate_limit_error", "slow down")).build()

    private fun internal(status: Int, type: String, vararg h: Pair<String, String>) = InternalServerException.builder()
        .statusCode(status).headers(headers(*h)).body(errorBody(type, "boom")).build()

    private fun badRequest(message: String) = BadRequestException.builder()
        .headers(headers()).body(errorBody("invalid_request_error", message)).build()

    private fun map(e: Throwable): PromptExecutionException = e.toPromptExecutionException(prompt)

    @Test
    fun `429 maps to quota exceeded with delay from Retry-After`() {
        val mapped = map(rateLimit("Retry-After" to "7"))
        assertIs<QuotaExceededException>(mapped)
        assertEquals(7.seconds, mapped.retryAfter)
    }

    @Test
    fun `429 prefers Retry-After-Ms and has no delay without headers`() {
        assertEquals(1500.milliseconds, assertIs<QuotaExceededException>(map(rateLimit("Retry-After-Ms" to "1500", "Retry-After" to "9"))).retryAfter)
        assertNull(assertIs<QuotaExceededException>(map(rateLimit())).retryAfter)
    }

    @Test
    fun `529 overloaded maps to transient server error`() {
        val e = internal(529, "overloaded_error", "Retry-After" to "3")
        val mapped = map(e)
        assertIs<TransientException>(mapped)
        assertEquals(TransientReason.SERVER_ERROR, mapped.reason)
        assertSame(e, mapped.cause)
        // The core TransientException has no delay field: the header value stays readable on the cause.
        assertEquals(3.seconds, retryAfterFrom(e))
        assertTrue(e.isOverloaded())
    }

    @Test
    fun `overloaded_error body on a non-529 status is still overloaded`() {
        assertTrue(internal(500, "overloaded_error").isOverloaded())
        assertFalse(internal(500, "api_error").isOverloaded())
    }

    @Test
    fun `other 5xx maps to transient server error`() {
        for (status in listOf(500, 502, 503)) {
            val mapped = map(internal(status, "api_error"))
            assertEquals(TransientReason.SERVER_ERROR, assertIs<TransientException>(mapped).reason)
        }
    }

    @Test
    fun `400 maps to invalid request`() {
        val mapped = map(badRequest("messages: text content blocks must be non-empty"))
        assertIs<InvalidRequestException>(mapped)
        assertEquals(400, mapped.httpCode)
        assertFalse(mapped.message.orEmpty().contains("workspace id"))
    }

    @Test
    fun `400 about anthropic-workspace-id carries the configuration hint`() {
        val mapped = map(badRequest("This API key is identity-linked: the anthropic-workspace-id header is required"))
        assertIs<InvalidRequestException>(mapped)
        assertEquals(400, mapped.httpCode)
        assertTrue(mapped.message.orEmpty().contains(WORKSPACE_ID_HINT), mapped.message)
    }

    @Test
    fun `auth and permission errors are fatal invalid requests`() {
        val unauthorized = UnauthorizedException.builder().headers(headers("x-api-key" to "sk-ant-secret")).body(errorBody("authentication_error", "invalid x-api-key")).build()
        val forbidden = PermissionDeniedException.builder().headers(headers()).body(errorBody("permission_error", "no access")).build()
        assertEquals(401, assertIs<InvalidRequestException>(map(unauthorized)).httpCode)
        assertEquals(403, assertIs<InvalidRequestException>(map(forbidden)).httpCode)
        assertFalse("sk-ant-secret" in map(unauthorized).message.orEmpty())
    }

    @Test
    fun `404 and 422 are fatal invalid requests`() {
        val notFound = NotFoundException.builder().headers(headers()).body(errorBody("not_found_error", "model")).build()
        val unprocessable = UnprocessableEntityException.builder().headers(headers()).body(errorBody("invalid_request_error", "bad")).build()
        assertEquals(404, assertIs<InvalidRequestException>(map(notFound)).httpCode)
        assertEquals(422, assertIs<InvalidRequestException>(map(unprocessable)).httpCode)
    }

    @Test
    fun `statuses without a dedicated class follow the SDK policy`() {
        fun unexpected(status: Int) = UnexpectedStatusCodeException.builder().statusCode(status).headers(headers()).body(errorBody("api_error", "x")).build()
        assertEquals(TransientReason.NETWORK_TIMEOUT, assertIs<TransientException>(map(unexpected(408))).reason)
        assertEquals(TransientReason.SERVER_ERROR, assertIs<TransientException>(map(unexpected(409))).reason)
        assertEquals(413, assertIs<InvalidRequestException>(map(unexpected(413))).httpCode)
        assertIs<UnknownProviderException>(map(unexpected(302)))
    }

    @Test
    fun `mid-stream error event is transient, overloaded or not`() {
        val overloaded = SseException.builder().statusCode(200).headers(headers()).body(errorBody("overloaded_error", "Overloaded")).build()
        val other = SseException.builder().statusCode(200).headers(headers()).body(errorBody("api_error", "broken")).build()
        assertEquals(TransientReason.SERVER_ERROR, assertIs<TransientException>(map(overloaded)).reason)
        assertEquals(TransientReason.NETWORK_ERROR, assertIs<TransientException>(map(other)).reason)
    }

    @Test
    fun `io errors and timeouts are transient`() {
        assertEquals(TransientReason.NETWORK_ERROR, assertIs<TransientException>(map(AnthropicIoException("reset"))).reason)
        assertEquals(TransientReason.NETWORK_TIMEOUT, assertIs<TransientException>(map(InterruptedIOException("t"))).reason)
        assertEquals(TransientReason.NETWORK_TIMEOUT, assertIs<TransientException>(map(IOException("Read timeout"))).reason)
        assertEquals(TransientReason.NETWORK_ERROR, assertIs<TransientException>(map(IOException("closed"))).reason)
    }

    @Test
    fun `already mapped exception is returned as is, unknown becomes UnknownProviderException`() {
        val mapped = TransientException(TransientReason.EMPTY_STREAM, "e", null)
        assertSame(mapped, map(mapped))
        assertIs<UnknownProviderException>(map(IllegalStateException("weird")))
    }

    @Test
    fun `retry strategy retries transient classes and never the fatal ones`() {
        val retry = listOf<Throwable>(
            rateLimit(), internal(500, "api_error"), internal(529, "overloaded_error"),
            SseException.builder().statusCode(200).headers(headers()).body(errorBody("overloaded_error", "o")).build(),
            AnthropicIoException("reset"),
            UnexpectedStatusCodeException.builder().statusCode(408).headers(headers()).body(errorBody("api_error", "x")).build(),
            UnexpectedStatusCodeException.builder().statusCode(409).headers(headers()).body(errorBody("api_error", "x")).build(),
            RuntimeException("wrapped", internal(503, "api_error")),
            IllegalStateException("Connection reset by peer"),
        )
        retry.forEach { assertTrue(AnthropicRetryStrategy.shouldRetry(it, 1), "must retry: $it") }

        val fatal = listOf<Throwable>(
            badRequest("anthropic-workspace-id required"),
            badRequest("bad"),
            UnauthorizedException.builder().headers(headers()).body(errorBody("authentication_error", "k")).build(),
            PermissionDeniedException.builder().headers(headers()).body(errorBody("permission_error", "p")).build(),
            NotFoundException.builder().headers(headers()).body(errorBody("not_found_error", "n")).build(),
            UnprocessableEntityException.builder().headers(headers()).body(errorBody("invalid_request_error", "u")).build(),
            IllegalStateException("something else"),
        )
        fatal.forEach { assertFalse(AnthropicRetryStrategy.shouldRetry(it, 1), "must not retry: $it") }
        assertTrue(fatal.filterIsInstance<AnthropicServiceException>().isNotEmpty())
    }
}
