package org.nobirds.murmuration.openai

import com.openai.models.responses.ResponseStatus
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.MaxTokensException
import org.nobirds.murmuration.prompt.PolicyCategory
import org.nobirds.murmuration.prompt.PolicyViolationException
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.UnknownProviderException
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.system
import org.nobirds.murmuration.prompt.text
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Unit tests for [mapResponseOutcome] — the Responses API terminal-status -> exception mapper
 * (the successor to chat completions' `finish_reason` mapping).
 */
class OpenAiResponseOutcomeTest {

    private val prompt: Prompt = buildPrompt {
        llm = Llm("azure", "gpt", "Test", "1", true, "gpt-5.4-nano")
        system { text("test") }
    }

    private fun map(status: ResponseStatus?, reason: String? = null, error: String? = null) =
        mapResponseOutcome(status, reason, error, prompt)

    @Test
    fun `completed is a normal completion`() {
        assertNull(map(ResponseStatus.COMPLETED))
    }

    @Test
    fun `null status is treated as completed`() {
        assertNull(map(null))
    }

    @Test
    fun `incomplete max_output_tokens maps to MaxTokensException`() {
        assertTrue(map(ResponseStatus.INCOMPLETE, reason = "max_output_tokens") is MaxTokensException)
    }

    @Test
    fun `incomplete content_filter maps to a prohibited policy violation`() {
        val error = map(ResponseStatus.INCOMPLETE, reason = "content_filter")
        assertTrue(error is PolicyViolationException)
        assertEquals(PolicyCategory.PROHIBITED, error.category)
    }

    @Test
    fun `incomplete for any other reason is a retryable transient`() {
        val error = map(ResponseStatus.INCOMPLETE, reason = "something_new")
        assertTrue(error is TransientException)
        assertEquals(TransientReason.INCOMPLETE_RESPONSE, error.reason)
    }

    @Test
    fun `failed with a content-filter error maps to a policy violation`() {
        val error = map(ResponseStatus.FAILED, error = "The response was blocked by the content_filter")
        assertTrue(error is PolicyViolationException)
    }

    @Test
    fun `failed with an opaque error maps to UnknownProviderException`() {
        assertTrue(map(ResponseStatus.FAILED, error = "internal boom") is UnknownProviderException)
    }

    @Test
    fun `in_progress at stream end is a retryable transient`() {
        val error = map(ResponseStatus.IN_PROGRESS)
        assertTrue(error is TransientException)
        assertEquals(TransientReason.INCOMPLETE_RESPONSE, error.reason)
    }
}
