package org.nobirds.murmuration.gemini

import com.google.genai.errors.ClientException
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.QuotaExceededException
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.retry.findServerRetryAfterSeconds
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.time.Duration.Companion.seconds

/** The "please retry in Ns" text hint and the typed retryAfter resolve to one value through [findServerRetryAfterSeconds]. */
class GeminiRetryAfterSingleSourceTest {

    private val prompt = Prompt(
        Llm("google", "gemini", "gemini-3-flash", "v1", true, "gemini-3-flash-preview"),
        null,
        listOf(PromptContent(PromptContentType.USER, listOf(TextPromptPart("hi")))),
    )

    @Test
    fun `text hint on a 429 ClientException becomes typed retryAfter and the shared function agrees`() {
        val mapped = ClientException(429, "RESOURCE_EXHAUSTED", "Quota hit. Please retry in 12.7s.")
            .toPromptExecutionException(prompt)
        assertIs<QuotaExceededException>(mapped)
        assertEquals(12.seconds, mapped.retryAfter)
        assertEquals(12, mapped.findServerRetryAfterSeconds())
    }

    @Test
    fun `message-only fallback path yields the same value as the typed path`() {
        val mapped = RuntimeException("resource exhausted, please retry in 12.7s").toPromptExecutionException(prompt)
        assertIs<QuotaExceededException>(mapped)
        assertEquals(12.seconds, mapped.retryAfter)
        assertEquals(12, QuotaExceededException(mapped.kind, 12.seconds, "x", null).findServerRetryAfterSeconds())
    }

    @Test
    fun `no hint gives null on both`() {
        val mapped = ClientException(429, "RESOURCE_EXHAUSTED", "Quota hit.").toPromptExecutionException(prompt)
        assertIs<QuotaExceededException>(mapped)
        assertEquals(null, mapped.retryAfter)
        assertEquals(null, mapped.findServerRetryAfterSeconds())
    }
}
