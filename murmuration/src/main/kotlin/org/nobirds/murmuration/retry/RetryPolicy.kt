package org.nobirds.murmuration.retry

import org.nobirds.murmuration.prompt.QuotaExceededException
import kotlin.random.Random
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

/**
 * Value object describing a complete retry behaviour: the [RetryStrategy] decision plus the
 * attempt budget and the exponential backoff bounds. Shared by every retry site in the
 * framework so the quadruple (strategy, maxRetries, backoff bounds) is not re-declared as
 * loose parameters at each of them.
 *
 * A server-supplied retry-after hint (typed `QuotaExceededException.retryAfter` or "please retry in Ns")
 * is honoured as given plus a 1 s margin, but capped at [MAX_SERVER_RETRY_AFTER] (120 s) so a huge hint
 * cannot park a round-trip for an hour; no jitter is added to it.
 *
 * Stateless and reusable: per run-of-attempts state (attempt counter, current backoff) lives
 * in a [RetrySession] created via [newSession].
 */
data class RetryPolicy(
    val strategy: RetryStrategy = DefaultRetryStrategy,
    /** How many retries are allowed after the initial attempt. */
    val maxRetries: Int = 3,
    val initialBackoff: Duration = 2.seconds,
    val maxBackoff: Duration = 10.minutes,
    /**
     * Pauses for quota errors ([QuotaExceededException], 429 / RESOURCE_EXHAUSTED) with no hint
     * from the server: the k-th consecutive quota attempt sleeps `quotaBackoffs[k-1]` (the last element is the ceiling).
     * A quota is cured by waiting, not by a quick retry, so the scale is steeper than the general backoff.
     */
    val quotaBackoffs: List<Duration> = listOf(15.seconds, 30.seconds, 60.seconds),
    /** Relative spread of the quota pause: ±[quotaJitter] (0.2 = ±20%). */
    val quotaJitter: Double = 0.2,
) {
    companion object {
        /** Upper bound of a server-supplied retry-after hint (see [RetryPolicy] KDoc); no jitter is added to it. */
        val MAX_SERVER_RETRY_AFTER: Duration = 120.seconds
    }

    init {
        require(quotaBackoffs.isNotEmpty()) { "quotaBackoffs must not be empty" }
    }

    fun newSession(random: Random = Random.Default): RetrySession = RetrySession(this, random)
}

/**
 * Mutable state of one run of attempts under a [RetryPolicy]: counts attempts, owns the
 * exponential backoff progression and honours provider-supplied retry-after hints
 * ([findServerRetryAfterSeconds]) over the computed backoff.
 *
 * Usage: on every failure call [nextDelayOrNull]; a non-null result is the pause to sleep
 * before the next attempt, `null` means the budget is exhausted (or the error is not
 * retryable under the strategy) and the error must propagate.
 *
 * Budgets are per error class: quota errors ([QuotaExceededException]) and everything else
 * (stream/network timeouts, 5xx, ...) each get their own [RetryPolicy.maxRetries]; 2×429 +
 * 2×timeout does not exhaust a shared budget. Quota pauses follow [RetryPolicy.quotaBackoffs]
 * with jitter unless the server supplied a retry-after.
 */
class RetrySession(private val policy: RetryPolicy, private val random: Random = Random.Default) {

    /** 1-based number of the failed attempt seen so far (all error classes together). */
    var attempt: Int = 0
        private set

    /**
     * 1-based number of the failed attempt INSIDE the budget of its error class (quota or other), as
     * of the last [nextDelayOrNull] call. This — not [attempt] — is what retry logs report next to
     * [maxRetries]: "2/2" is possible, "4/2" is not.
     */
    var attemptInBudget: Int = 0
        private set

    /** Budget from the owning policy — exposed for logging at retry sites. */
    val maxRetries: Int get() = policy.maxRetries

    private var currentBackoff: Duration = policy.initialBackoff

    private var quotaAttempts: Int = 0
    private var otherAttempts: Int = 0

    /**
     * The pause before the next attempt, or `null` when [error] must propagate (abort, exhausted
     * budget of its class, strategy refusal). [bypassStrategy] — the caller has already decided that
     * this failure is retryable (a per-attempt timeout of [retryWithTimeout]); the budget and the
     * abort check still apply.
     */
    fun nextDelayOrNull(error: Throwable, bypassStrategy: Boolean = false): Duration? {
        attempt++
        val quota = error.causes.filterIsInstance<QuotaExceededException>().firstOrNull()
        val classAttempts = if (quota != null) ++quotaAttempts else ++otherAttempts
        attemptInBudget = classAttempts
        if (error.abortsAgent || classAttempts > policy.maxRetries) return null
        if (!bypassStrategy && !policy.strategy.shouldRetry(error, attempt)) return null

        val retryAfterSeconds = (quota ?: error).findServerRetryAfterSeconds()
        if (retryAfterSeconds != null) {
            // Provider told us when to come back — trust it (plus a safety margin) and do
            // not escalate the local backoff: the hint already reflects the server state.
            return retryAfterSeconds.seconds.coerceAtMost(RetryPolicy.MAX_SERVER_RETRY_AFTER) + 1.seconds
        }

        if (quota != null) {
            val base = policy.quotaBackoffs[(classAttempts - 1).coerceAtMost(policy.quotaBackoffs.lastIndex)]
            val factor = 1.0 + (random.nextDouble() * 2 - 1) * policy.quotaJitter
            return base * factor
        }

        val delay = currentBackoff
        currentBackoff = (currentBackoff * 2).coerceAtMost(policy.maxBackoff) + (random.nextDouble() * 1000).milliseconds
        return delay
    }
}

/**
 * The server's EXPLICIT retry-after hint: the typed [QuotaExceededException.retryAfter] or
 * `"please retry in 12.3s"` in the message. The one place that answers "how long does the server say
 * to wait"; there is no default for "the text says 429" — a quota without a hint goes onto
 * [RetryPolicy.quotaBackoffs], and a caller that needs a default names it itself.
 */
fun Throwable.findServerRetryAfterSeconds(): Int? {
    if (this is QuotaExceededException) {
        retryAfter?.inWholeSeconds?.toInt()?.let { return it }
    }
    val message = message?.lowercase() ?: return null
    val retryAfterMatch = Regex("please retry in (\\d+(\\.\\d+)?)s").find(message)
    return retryAfterMatch?.groupValues?.get(1)?.toDoubleOrNull()?.toInt()
}
