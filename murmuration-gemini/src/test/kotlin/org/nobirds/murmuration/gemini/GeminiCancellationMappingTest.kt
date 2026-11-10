package org.nobirds.murmuration.gemini

import com.google.genai.errors.ClientException
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.InvalidRequestException
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.retry.DefaultRetryStrategy
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * Fix 1: a Gemini 499 "operation was cancelled" (the request aborted because an external
 * round-trip ceiling elapsed before the model replied) must map to a retryable
 * [TransientException], NOT the non-retryable [InvalidRequestException] the generic client-error
 * path used to produce. That miss was the root of the hung round-trip: a 499 after ~598s with no
 * retry.
 */
class GeminiCancellationMappingTest {

    private val prompt = Prompt(
        Llm("google", "gemini", "gemini-3-flash", "v1", true, "gemini-3-flash-preview"),
        null,
        listOf(PromptContent(PromptContentType.USER, listOf(TextPromptPart("hi")))),
    )

    // ApiException(int code, String status, String message) → getMessage() == "<code> <status>. <message>"
    private fun clientException(code: Int, status: String, message: String) =
        ClientException(code, status, message)

    @Test
    fun `499 ClientException maps to retryable TransientException NETWORK_TIMEOUT`() {
        val mapped = clientException(499, "CANCELLED", "The operation was cancelled.")
            .toPromptExecutionException(prompt)

        val transient = assertIs<TransientException>(mapped, "499 must map to TransientException")
        assertEquals(TransientReason.NETWORK_TIMEOUT, transient.reason)
        assertTrue(
            DefaultRetryStrategy.shouldRetry(transient, attempt = 1),
            "DefaultRetryStrategy must retry the 499-derived TransientException",
        )
    }

    @Test
    fun `operation was cancelled in a raw exception maps to retryable TransientException`() {
        // The cancellation can also reach the mapper wrapped in an untyped RuntimeException that
        // bypassed the SDK's ClientException (side-channel paths). The message-based fallback
        // (mapByMessage) must still route it to the retryable path.
        val mapped = RuntimeException("io.grpc.StatusRuntimeException: operation was cancelled")
            .toPromptExecutionException(prompt)

        val transient = assertIs<TransientException>(mapped)
        assertEquals(TransientReason.NETWORK_TIMEOUT, transient.reason)
        assertTrue(DefaultRetryStrategy.shouldRetry(transient, attempt = 1))
    }

    @Test
    fun `plain 400 still maps to non-retryable InvalidRequestException (regression guard)`() {
        val mapped = clientException(400, "INVALID_ARGUMENT", "bad request")
            .toPromptExecutionException(prompt)

        assertIs<InvalidRequestException>(mapped, "generic 400 must stay a non-retryable InvalidRequestException")
        assertFalse(
            DefaultRetryStrategy.shouldRetry(mapped, attempt = 1),
            "a plain InvalidRequestException must NOT be retried",
        )
    }
}
