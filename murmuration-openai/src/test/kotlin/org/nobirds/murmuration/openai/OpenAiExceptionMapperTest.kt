package org.nobirds.murmuration.openai

import com.openai.core.http.Headers
import com.openai.errors.BadRequestException
import com.openai.errors.InternalServerException
import com.openai.errors.OpenAIIoException
import com.openai.errors.RateLimitException
import com.openai.errors.UnauthorizedException
import com.openai.models.ErrorObject
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.InvalidRequestException
import org.nobirds.murmuration.prompt.PolicyCategory
import org.nobirds.murmuration.prompt.PolicyViolationException
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.QuotaExceededException
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.system
import org.nobirds.murmuration.prompt.text
import java.io.IOException
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds

/**
 * Unit tests for [Throwable.toPromptExecutionException] — the Azure OpenAI SDK exception ->
 * provider-neutral [org.nobirds.murmuration.prompt.PromptExecutionException] mapper. SDK
 * exceptions are constructed via their own builders (the real exception types, not mocks) so
 * the assertions exercise our mapping logic against the actual shapes the SDK produces.
 */
class OpenAiExceptionMapperTest {

    private val prompt: Prompt = buildPrompt {
        llm = Llm("azure", "gpt", "Test", "1", true, "gpt-5.4-nano")
        system { text("test") }
    }

    @Test
    fun `rate limit maps to quota exceeded with retry-after from the header`() {
        val error = RateLimitException.builder()
            .headers(Headers.builder().put("Retry-After", "7").build())
            .error(ErrorObject.builder().code("rate_limit_exceeded").message("Rate limit exceeded").type("rate_limit").param("").build())
            .build()

        val mapped = error.toPromptExecutionException(prompt)
        assertTrue(mapped is QuotaExceededException)
        assertEquals(7.seconds, mapped.retryAfter)
    }

    @Test
    fun `bad request carrying a content management policy marker is a policy violation`() {
        val error = BadRequestException.builder()
            .headers(Headers.builder().build())
            .error(
                ErrorObject.builder()
                    .code("content_filter")
                    .message("The response was filtered due to the prompt triggering Azure OpenAI's content management policy")
                    .type("invalid_request_error").param("")
                    .build()
            )
            .build()

        val mapped = error.toPromptExecutionException(prompt)
        assertTrue(mapped is PolicyViolationException)
        assertEquals(PolicyCategory.PROHIBITED, mapped.category)
    }

    @Test
    fun `gpt-image safety rejection maps to a policy violation with the parsed safety category`() {
        // Verbatim shape observed from a live gpt-image rejection (pipeline a39c23bf):
        // typed code "moderation_blocked" + a `safety_violations=[...]` marker in the message.
        val error = BadRequestException.builder()
            .headers(Headers.builder().build())
            .error(
                ErrorObject.builder()
                    .code("moderation_blocked")
                    .message(
                        "Your request was rejected by the safety system. Your prompt may contain content " +
                            "that is not allowed by our safety system. safety_violations=[sexual]."
                    )
                    .type("image_generation_user_error").param("")
                    .build()
            )
            .build()

        val mapped = error.toPromptExecutionException(prompt)
        assertTrue(mapped is PolicyViolationException)
        assertEquals(PolicyCategory.SAFETY, mapped.category)
        assertEquals("sexual", mapped.finishReason)
    }

    @Test
    fun `moderation_blocked code alone classifies as a safety policy violation even without a parseable message`() {
        val error = BadRequestException.builder()
            .headers(Headers.builder().build())
            .error(
                ErrorObject.builder()
                    .code("moderation_blocked")
                    .message("Request blocked for an unspecified reason")
                    .type("image_generation_user_error").param("")
                    .build()
            )
            .build()

        val mapped = error.toPromptExecutionException(prompt)
        assertTrue(mapped is PolicyViolationException)
        assertEquals(PolicyCategory.SAFETY, mapped.category)
        assertEquals("moderation_blocked", mapped.finishReason)
    }

    @Test
    fun `content management policy marker without a typed code still falls back to a policy violation`() {
        val error = BadRequestException.builder()
            .headers(Headers.builder().build())
            .error(
                ErrorObject.builder()
                    .code("unexpected_code")
                    .message("The response was filtered due to the prompt triggering Azure OpenAI's content management policy")
                    .type("invalid_request_error").param("")
                    .build()
            )
            .build()

        val mapped = error.toPromptExecutionException(prompt)
        assertTrue(mapped is PolicyViolationException)
        assertEquals(PolicyCategory.PROHIBITED, mapped.category)
    }

    @Test
    fun `other bad requests are invalid request errors`() {
        val error = BadRequestException.builder()
            .headers(Headers.builder().build())
            .error(ErrorObject.builder().code("unknown_parameter").message("Unknown parameter: foo").type("invalid_request_error").param("").build())
            .build()

        val mapped = error.toPromptExecutionException(prompt)
        assertTrue(mapped is InvalidRequestException)
        assertEquals(400, mapped.httpCode)
    }

    @Test
    fun `unauthorized triggers an auth-refresh retry`() {
        val error = UnauthorizedException.builder()
            .headers(Headers.builder().build())
            .error(ErrorObject.builder().code("invalid_api_key").message("Invalid API key").type("invalid_request_error").param("").build())
            .build()

        val mapped = error.toPromptExecutionException(prompt)
        assertTrue(mapped is TransientException)
        assertEquals(TransientReason.AUTH_REFRESH, mapped.reason)
    }

    @Test
    fun `internal server error is a transient server error`() {
        val error = InternalServerException.builder()
            .statusCode(500)
            .headers(Headers.builder().build())
            .error(ErrorObject.builder().code("internal_error").message("Internal error").type("server_error").param("").build())
            .build()

        val mapped = error.toPromptExecutionException(prompt)
        assertTrue(mapped is TransientException)
        assertEquals(TransientReason.SERVER_ERROR, mapped.reason)
    }

    @Test
    fun `an OpenAIIoException is a transient network error`() {
        val error = OpenAIIoException("Connection failed", IOException("boom"))

        val mapped = error.toPromptExecutionException(prompt)
        assertTrue(mapped is TransientException)
        assertEquals(TransientReason.NETWORK_ERROR, mapped.reason)
    }

    @Test
    fun `an already-mapped exception is returned as-is`() {
        val already = TransientException(TransientReason.NETWORK_TIMEOUT, "timeout", null)

        val mapped = already.toPromptExecutionException(prompt)
        assertSame(already, mapped)
    }
}
