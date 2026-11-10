package org.nobirds.murmuration.retry

import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.InvalidRequestException
import org.nobirds.murmuration.prompt.MaxTokensException
import org.nobirds.murmuration.prompt.PolicyCategory
import org.nobirds.murmuration.prompt.PolicyViolationException
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Unit tests for [DefaultRetryStrategy] — the provider-neutral retry decision. Focus on the
 * bounded retry of [PolicyViolationException] (safety / content-filter block): the filter is
 * stochastic, so a block gets a small number of retries and only then fails honestly.
 */
class DefaultRetryStrategyTest {

    private val prompt: Prompt = Prompt(
        Llm("test", "test", "test", "v1", true, "test-model"),
        settings = null,
        content = emptyList(),
    )

    private fun policy(category: PolicyCategory = PolicyCategory.PROHIBITED) =
        PolicyViolationException(prompt, category, "content_filter", "Blocked", null)

    @Test
    fun `policy violation is retried while under the cap`() {
        val e = policy()
        assertTrue(DefaultRetryStrategy.shouldRetry(e, attempt = 1))
        assertTrue(DefaultRetryStrategy.shouldRetry(e, attempt = DefaultRetryStrategy.POLICY_VIOLATION_MAX_RETRIES))
    }

    @Test
    fun `policy violation becomes fatal once the cap is exceeded`() {
        val e = policy()
        assertFalse(DefaultRetryStrategy.shouldRetry(e, attempt = DefaultRetryStrategy.POLICY_VIOLATION_MAX_RETRIES + 1))
    }

    @Test
    fun `bounded retry applies to every policy category, provider-neutral`() {
        PolicyCategory.entries.forEach { category ->
            assertTrue(
                DefaultRetryStrategy.shouldRetry(policy(category), attempt = 1),
                "category $category should be retried while under the cap",
            )
        }
    }

    @Test
    fun `transient failures ignore the attempt count`() {
        val e = TransientException(TransientReason.NETWORK_ERROR, "boom", null)
        assertTrue(DefaultRetryStrategy.shouldRetry(e, attempt = 1))
        assertTrue(DefaultRetryStrategy.shouldRetry(e, attempt = 99))
    }

    @Test
    fun `invalid request is never retried`() {
        val e = InvalidRequestException(prompt, httpCode = 400, message = "bad", cause = null)
        assertFalse(DefaultRetryStrategy.shouldRetry(e, attempt = 1))
    }

    @Test
    fun `max tokens is retried`() {
        val e = MaxTokensException(prompt, "truncated", null)
        assertTrue(DefaultRetryStrategy.shouldRetry(e, attempt = 1))
    }
}
