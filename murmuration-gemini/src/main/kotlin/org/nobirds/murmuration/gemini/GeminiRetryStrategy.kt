package org.nobirds.murmuration.gemini

import com.google.genai.errors.ClientException
import com.google.genai.errors.ServerException
import org.nobirds.murmuration.retry.RetryStrategy
import org.nobirds.murmuration.retry.causes

/**
 * Gemini-specific retry triggers: HTTP cases from `com.google.genai.errors` (429, 5xx)
 * plus substring matching on messages the SDK returns without a structural exception:
 * - `Cache content … is expired` — context cache expired (Gemini 400; retrying is safe,
 *   a fresh cache will be created).
 * - `mint used too late` / `GaiaMint invalid` — Google's internal auth mint expired
 *   between requests; on retry the SDK refreshes the token itself.
 *
 * Compose this with [org.nobirds.murmuration.retry.DefaultRetryStrategy] using
 * `DefaultRetryStrategy or GeminiRetryStrategy` to get the standard "core + Gemini"
 * coverage that application code typically wants.
 */
object GeminiRetryStrategy : RetryStrategy {
    private val MESSAGES = listOf(
        // Gemini context cache: 400 "Cache content ... is expired". Retry-safe: a fresh
        // cache is created on retry.
        "cache content", "is expired", "cache expired",
        // Google internal auth (UberMint/GaiaMint): the mint lives ~6 minutes and may
        // expire between requests. On retry the SDK refreshes the token through
        // GoogleCredentials.refresh().
        "mint used too late", "gaiamint invalid",
    )

    override fun shouldRetry(throwable: Throwable, attempt: Int): Boolean =
        throwable.causes.any { it.matches() }

    private fun Throwable.matches(): Boolean {
        val msg = message?.lowercase() ?: ""
        return when (this) {
            is ClientException if code() == 429 -> true
            is ServerException if code() >= 500 -> true
            else -> MESSAGES.any { it in msg }
        }
    }
}
