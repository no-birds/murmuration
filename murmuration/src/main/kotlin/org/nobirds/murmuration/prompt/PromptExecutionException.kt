package org.nobirds.murmuration.prompt

import kotlin.time.Duration

/**
 * Provider-neutral model of prompt-execution failures. Every exception that crosses the
 * boundary of a provider wrapper (e.g. `:murmuration-gemini`) must be a subtype of this
 * class — the provider mapper translates SDK-specific exceptions into one of these on its
 * boundary. Downstream consumers (`RetryStrategy`, `RecoveryAgent`, statistics persistence,
 * UI/DB) work with these types instead of parsing message strings.
 *
 * The hierarchy is two-level:
 * - **Root** [PromptExecutionException] — failures not tied to a specific prompt (network,
 *   quota, cache, malformed response, unknown provider error).
 * - **Intermediate** [PromptRelatedException] — failures the provider associated with a
 *   specific [Prompt] (safety/policy block, max-tokens, invalid request). Subclasses carry
 *   the prompt directly so [RecoveryAgent]-style logic and logging can read it without
 *   side-channels.
 *
 * Formal codes:
 * - Multi-case types ([TransientException], [QuotaExceededException], [PolicyViolationException],
 *   [MalformedResponseException]) carry an enum field that classifies the cause as far as
 *   the provider's response allows.
 * - Single-case types ([CacheExpiredException], [MaxTokensException]) use the class itself
 *   as the discriminator.
 * - [InvalidRequestException] uses the HTTP status code; [UnknownProviderException] uses the
 *   FQN of the original provider exception.
 *
 * `cause` is always passed explicitly: real SDK exceptions get wrapped, synthetic cases (our
 * idle stream timeout, empty stream) get `null`. The argument has no default so a missing
 * cause is visible at the call site.
 */
sealed class PromptExecutionException(message: String, cause: Throwable?) : RuntimeException(message, cause) {
    /**
     * Stream diagnostics captured at the failure point when the failure happened during a
     * streamed response. The provider executor sets this before the exception propagates, so the
     * failure statistics — rebuilt downstream from the exception, not from the discarded
     * success-path aggregate — can still persist it. Null for non-streamed failures.
     */
    var streamDiagnostics: StreamDiagnostics? = null

    /**
     * Partial [PromptExecutionStatistics] the round-trip had already assembled before it broke —
     * tokens burned, the response message received so far (tool calls included), model name,
     * and everything else a normal completion would carry. The consumer that owns the
     * round-trip's assembly ([aggregate] and
     * [org.nobirds.murmuration.tooling.ToolCallingAgent]'s turn collector) attaches it here
     * before rethrowing, so failure statistics rebuilt downstream — from the exception, not from
     * the discarded success-path aggregate — carry the round-trip's real partial content instead
     * of an empty stub. Null only when the executor never reached
     * the request (no `PromptStarted` — it is emitted before the request is sent).
     *
     * Widened from a bare token total (`partialUsage`) because a broken stream that already
     * received `tool_call` parts has more to lose than tokens: without this, a FAILURE record
     * whose tools actually ran had no `responseMessage` to show for it.
     */
    var partialStats: PromptExecutionStatistics? = null
}

/**
 * A failure tied to a specific [prompt] — safety/policy rejection, MAX_TOKENS, provider-side
 * malformed request. A [RecoveryAgent]-style handler reads [prompt] and produces a new one;
 * the next attempt uses the rewritten content.
 *
 * Failures that "happened during prompt execution" but are not caused by the prompt itself
 * — transient network/server problems, quota, cache expiry, malformed model response — are
 * flat subclasses of [PromptExecutionException] without a `prompt` field.
 */
sealed class PromptRelatedException(
    val prompt: Prompt,
    message: String,
    cause: Throwable?,
) : PromptExecutionException(message, cause)

/**
 * Transient infrastructure failure: retry as-is without changing the prompt. Covers network
 * timeouts/breaks, server-side 5xx, auth-token refreshes, empty streams.
 */
class TransientException(
    val reason: TransientReason,
    message: String,
    cause: Throwable?,
) : PromptExecutionException(message, cause)

enum class TransientReason {
    /** Per-attempt timeout, idle stream timeout, socket timeout. */
    NETWORK_TIMEOUT,

    /** Connection reset / canceled / other I/O breaks. */
    NETWORK_ERROR,

    /** Provider returned 5xx. */
    SERVER_ERROR,

    /** Auth token / mint expired between requests; SDK refreshes on retry. */
    AUTH_REFRESH,

    /** Stream finished without yielding a single chunk. */
    EMPTY_STREAM,

    /**
     * Stream produced chunks but ended without a terminating `finishReason` (no STOP, no
     * MAX_TOKENS, no SAFETY, nothing). Observed with Gemini flash-preview after long
     * thinking phases — server drops the SSE connection mid-generation. Retry usually
     * completes normally.
     */
    INCOMPLETE_RESPONSE,

    /**
     * The provider answered 200 and finished normally, but the response carries nothing: no text, no
     * thought, no tool call. An empty answer expresses no intention of the model (neither "continue"
     * nor "done"), so it is a provider error, not an agent turn. Observed with Claude at high effort;
     * a fresh attempt usually answers.
     */
    EMPTY_RESPONSE,
}

/**
 * Rate limit / quota exceeded. Retryable, but with a delay of [retryAfter] when the provider
 * returned a `Retry-After` header (otherwise the retry policy decides). [RetryStrategy] reads
 * [retryAfter] to compute the per-attempt sleep.
 */
class QuotaExceededException(
    val kind: QuotaKind,
    val retryAfter: Duration?,
    message: String,
    cause: Throwable?,
) : PromptExecutionException(message, cause)

enum class QuotaKind {
    /** Per-second / per-minute rate limit. */
    RATE_LIMIT,

    /** Daily or project-wide quota (RESOURCE_EXHAUSTED). */
    RESOURCE_EXHAUSTED,

    /** Provider returned 429 without a clear category. */
    UNSPECIFIED,
}

/**
 * Provider-side cache expired (Gemini Context Cache TTL). Safe to retry — the wrapper will
 * recreate the cache on the next attempt. [cacheId] is filled when it can be extracted from
 * the response.
 */
class CacheExpiredException(
    val cacheId: String?,
    message: String,
    cause: Throwable?,
) : PromptExecutionException(message, cause)

/**
 * Provider rejected the request or response on a safety/content-policy basis. Retrying the
 * same prompt deterministically yields the same outcome — the only fix is a prompt rewrite
 * (`RecoveryAgent`) or a fallback. [finishReason] preserves the raw provider token (e.g.
 * `IMAGE_PROHIBITED_CONTENT`) for tracing and for forward-compatibility with provider values
 * not yet reflected in [PolicyCategory].
 */
class PolicyViolationException(
    prompt: Prompt,
    val category: PolicyCategory,
    val finishReason: String?,
    message: String,
    cause: Throwable?,
) : PromptRelatedException(prompt, message, cause)

enum class PolicyCategory {
    /** Gemini SAFETY / IMAGE_SAFETY. */
    SAFETY,

    /** PROHIBITED_CONTENT / IMAGE_PROHIBITED_CONTENT / BLOCKLIST. */
    PROHIBITED,

    /** Recitation of training-set text (RECITATION / IMAGE_RECITATION). */
    RECITATION,

    /** Sensitive PII. */
    SPII,

    /** OTHER / IMAGE_OTHER — provider did not specify. */
    OTHER,
}

/**
 * Provider response is structurally broken: unparseable JSON, malformed function-call,
 * missing expected payload section. Not tied to [Prompt] directly — it's a response-side
 * failure.
 *
 * Marked retryable in [DefaultRetryStrategy] (since 2026-05-27). Function-call sampling is
 * non-deterministic — each attempt independently produces its JSON; a retry on the same
 * prompt often picks a syntactically/schematically valid one. 3 attempts via RetryAgent;
 * if all 3 still produce malformed output it's a real prompt/schema issue and the caller
 * should inspect prompt/schema rather than keep retrying.
 */
class MalformedResponseException(
    val phase: ParsePhase,
    message: String,
    cause: Throwable?,
) : PromptExecutionException(message, cause)

enum class ParsePhase {
    /** Failed to deserialize payload (kotlinx.serialization). */
    SERIALIZATION,

    /** finish_reason = MALFORMED_FUNCTION_CALL. */
    FUNCTION_CALL,

    /** finish_reason = UNEXPECTED_TOOL_CALL, or the model called a tool that doesn't exist. */
    TOOL_CALL,

    /** Image response without an inline-data blob. */
    IMAGE_BLOB_MISSING,

    /** Embedding response without any embedding vector. */
    EMBEDDING_MISSING,
}

/**
 * The model hit the token limit and truncated the response (`finish_reason = MAX_TOKENS`).
 * The response is not malformed — just cut.
 *
 * Marked retryable in [DefaultRetryStrategy] (since 2026-05-27). Rationale: thinking-enabled
 * generation is **non-deterministic** — a retry on the same prompt often picks a different
 * thinking trajectory and fits the cap. We give it 3 shots (RetryAgent default); if all 3 hit
 * MAX_TOKENS it's a real loop / oversized task and the caller's recovery (raise cap, shorten
 * input, compress prompt via [PromptRelatedException.prompt]) takes over.
 */
class MaxTokensException(
    prompt: Prompt,
    message: String,
    cause: Throwable?,
) : PromptRelatedException(prompt, message, cause)

/**
 * The request is invalid on the provider side: bad params, unsupported language, forbidden,
 * etc. Retrying is pointless — caller has to fix [prompt] or settings. [httpCode] is the
 * formal provider code.
 */
class InvalidRequestException(
    prompt: Prompt,
    val httpCode: Int?,
    message: String,
    cause: Throwable?,
) : PromptRelatedException(prompt, message, cause)

/**
 * Provider threw something unexpected that doesn't map into any known case. [rawClass] is
 * the FQN of the original exception, kept for tracing. Encountering this in logs is a signal
 * to extend the mapper.
 */
class UnknownProviderException(
    val rawClass: String,
    message: String,
    cause: Throwable?,
) : PromptExecutionException(message, cause)

/**
 * Formal sub-classification of a [PromptExecutionException] as a stable string token,
 * suitable for storing in statistics columns and grouping in BI queries. Returns the inner
 * enum value's name for multi-case types and a fixed token for single-case types.
 *
 * Examples: `"PROHIBITED"` for `PolicyViolationException(category=PROHIBITED)`,
 * `"NETWORK_TIMEOUT"` for `TransientException(reason=NETWORK_TIMEOUT)`, `"CACHE_EXPIRED"`
 * for `CacheExpiredException`. The exception type itself is also useful for queries —
 * pair this with `e::class.simpleName` if needed.
 */
fun PromptExecutionException.errorCode(): String = when (this) {
    is PolicyViolationException -> category.name
    is TransientException -> reason.name
    is QuotaExceededException -> kind.name
    is MalformedResponseException -> phase.name
    is CacheExpiredException -> "CACHE_EXPIRED"
    is MaxTokensException -> "MAX_TOKENS"
    is InvalidRequestException -> "INVALID_REQUEST"
    is UnknownProviderException -> "UNKNOWN"
}

/**
 * Provider's raw `finish_reason` token if this exception carries one (currently only
 * [PolicyViolationException]). Returns `null` otherwise.
 */
fun PromptExecutionException.finishReasonOrNull(): String? =
    (this as? PolicyViolationException)?.finishReason
