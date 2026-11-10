package org.nobirds.murmuration.prompt

import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.emitAll
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import org.nobirds.murmuration.internal.logger
import org.nobirds.murmuration.retry.findServerRetryAfterSeconds
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

/**
 * `PromptExecutor` decorator that limits the frequency of image API calls. Applies only
 * to prompts with `ImagePromptSettings` — text calls pass through without rate limiting.
 *
 * Image preview models (Gemini/Vertex Imagen etc.) limit not only concurrency,
 * but also **RPM** (requests per minute). A semaphore alone is not enough: even a strictly serial
 * queue yields 6–12 RPM and runs into 429. So besides [maxConcurrency] the decorator
 * enforces at least [minIntervalBetweenRequests] **after** each request (successful
 * or not) — the next request cannot start earlier.
 *
 * On 429 / "Please retry in Xs" the delay is taken from the response, but not less than
 * [minIntervalBetweenRequests].
 */
class ThrottledImagePromptExecutor(
    private val delegate: PromptExecutor,
    maxConcurrency: Int = 1,
    private val minIntervalBetweenRequests: Duration = 5.seconds,
) : PromptExecutor {

    private val logger = logger()
    private val semaphore = Semaphore(maxConcurrency)

    private companion object {
        val DEFAULT_QUOTA_PAUSE: Duration = 10.seconds
    }

    override fun execute(prompt: Prompt): Flow<PromptExecutionEvent> {
        if (prompt.settings !is ImagePromptSettings) {
            return delegate.execute(prompt)
        }
        return flow {
            semaphore.withPermit {
                var failure: Throwable? = null
                try {
                    emitAll(delegate.execute(prompt))
                } catch (e: Throwable) {
                    failure = e
                }
                val pause = computePause(failure)
                if (pause > Duration.ZERO) delay(pause)
                failure?.let { throw it }
            }
        }
    }

    /**
     * The server's explicit retry-after hint; a failure that only SAYS it is a quota error
     * ("429" / "resource exhausted") without a hint gets [DEFAULT_QUOTA_PAUSE]. The throttler's own
     * default, named here — it is not a retry policy (the throttler never retries).
     */
    private fun Throwable.serverOrDefaultRetryAfter(): Duration? {
        findServerRetryAfterSeconds()?.let { return it.seconds }
        val text = message?.lowercase() ?: return null
        return if (text.contains("resource exhausted") || text.contains("429")) DEFAULT_QUOTA_PAUSE else null
    }

    private fun computePause(failure: Throwable?): Duration {
        if (failure == null) return minIntervalBetweenRequests
        val suggested = failure.serverOrDefaultRetryAfter()
        val pause = suggested?.coerceAtLeast(minIntervalBetweenRequests) ?: minIntervalBetweenRequests
        if (suggested != null) {
            logger.warn(
                "Image API requested retry-after {} (using {}). Cause: {}",
                suggested, pause, failure.message
            )
        }
        return pause
    }
}

/**
 * Wraps this [PromptExecutor] in a [ThrottledImagePromptExecutor] — image prompts
 * pass through the rate limit, the rest pass through without delay.
 */
fun PromptExecutor.imageThrottled(
    maxConcurrency: Int = 1,
    minIntervalBetweenRequests: Duration = 5.seconds,
): PromptExecutor =
    ThrottledImagePromptExecutor(this, maxConcurrency, minIntervalBetweenRequests)
