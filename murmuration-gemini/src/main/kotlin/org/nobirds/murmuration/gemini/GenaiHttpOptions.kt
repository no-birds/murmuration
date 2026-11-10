package org.nobirds.murmuration.gemini

import com.google.genai.types.HttpOptions
import com.google.genai.types.HttpRetryOptions

/**
 * Client-level [HttpOptions] for production genai clients: the SDK's built-in retry is switched off.
 *
 * The SDK installs `RetryInterceptor` on EVERY request (generateContent, generateContentStream,
 * caches.create/update, ...) and, with no `retryOptions` set, retries 408/429/5xx up to 5 attempts with
 * exponential backoff (1 s base, 60 s cap). That is a second retry layer under our own `RetrySession` /
 * quota ladder: one logical attempt of ours would hide up to 5 HTTP requests and tens of seconds of
 * silent backoff inside the TTFB window. `attempts(1)` leaves exactly one HTTP request per attempt;
 * all retry decisions are ours. HTTP timeouts are untouched.
 */
fun HttpOptions.Builder.withoutSdkRetry(): HttpOptions.Builder =
    retryOptions(HttpRetryOptions.builder().attempts(1).build())
