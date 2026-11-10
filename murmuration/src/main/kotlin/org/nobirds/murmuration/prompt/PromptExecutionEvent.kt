package org.nobirds.murmuration.prompt

import kotlin.time.Duration

/**
 * Delta event from a prompt-execution stream. Producer (executor) emits events as soon as
 * the underlying provider produces them — and *only* primitives of what happened. There is
 * no «complete statistics snapshot» event; the final [PromptExecutionStatistics] is
 * assembled by [aggregate] from the event sequence:
 *
 * - **[PromptStarted]** — fired once, BEFORE the request is sent to the provider, with
 *   everything known up-front: requested model name, request payload, an optional `cacheId`
 *   if a context cache is being reused. A round-trip that dies before the provider answers
 *   (TTFB timeout, connection error) still carries this data out.
 * - **[StreamResponded]** — fired once, at the first sign of a provider answer, never
 *   before [PromptStarted]. Carries the model name the provider actually served.
 * - **[PartReceived]** — one per [MessagePart] the model produces (text / thought /
 *   tool-call / inline-data / file). Wraps a sealed [MessagePart] so consumers
 *   pattern-match on `event.part` when they care.
 * - **[UsageDelta]** — token-usage delta. Both `input` and `output` are increments since the
 *   previous delta — aggregator sums them up. Providers with cumulative usage (Gemini) emit
 *   deltas by subtracting their previous snapshot. Input arrives as deltas too: some providers
 *   (Gemini streaming) only fill `usageMetadata.promptTokenCount` in the last chunk while
 *   `cachedContentTokenCount` is known up-front — that just means the executor emits the
 *   cached portion early and the non-cached portion at the end. Aggregator doesn't care
 *   about the timing, only sums.
 * - **[CacheCreated]** — a context cache was created as a separate API call alongside this
 *   round-trip. Side-channel event consumed by the agent via `Flow.onEach` to report an
 *   extra `llm_statistics` entry under `nature = CACHE_CREATE`. Aggregator ignores it.
 * - **[Completed]** / **[Failed]** — terminal events. Carry `finishReason` so downstream
 *   can tell e.g. a `PolicyViolation`-marked completion from a clean stop.
 *
 * Why one envelope for content: [MessagePart] is already a sealed hierarchy that captures
 * the meaningful structural distinction. Splitting the same distinction across event types
 * as well meant duplicate matching and made adding a new part type cost two places instead
 * of one.
 */
sealed interface PromptExecutionEvent {

    /**
     * Initial event of a prompt execution, emitted BEFORE the request is sent — carries the
     * per-call metadata known up-front: which model was selected, the full prompt as sent,
     * attempts so far, and (optionally) a reused `cacheId` plus its statistics. **Token counts do not live here** — both input
     * and output flow through [UsageDelta], even when input is technically known earlier.
     * Keeping all usage in one channel removes timing footguns (Gemini streaming, for one,
     * fills `promptTokenCount` only in the last chunk).
     */
    data class PromptStarted(
        val modelName: String,
        val requestedModelName: String?,
        val requestMessages: List<PromptMessage>,
        val cacheId: String? = null,
        val cacheStats: PromptCachingProvider.CacheStatistics? = null,
        val attempts: Int = 1,
    ) : PromptExecutionEvent

    /**
     * The provider answered: fired exactly once per round-trip, on the first event of the
     * provider's response of ANY type, after [PromptStarted] and before every other event except
     * [PromptStarted] (so before [UsageDelta], [PartReceived] and the terminal event). The single
     * definition for every provider — Gemini stream: the first chunk; Gemini direct / OpenAI
     * direct: the response is received; OpenAI stream: the first stream event; single-shot
     * replays: right after [PromptStarted].
     *
     * [servedModelName] is best-effort, a documented property of the field: Gemini takes the
     * response's `modelVersion`; an OpenAI stream takes the model from `response.created` only
     * if that is the first event, otherwise the requested model (and it is not updated later),
     * because the Azure/proxy transport may open the stream with another event. It is the only
     * source of `PromptExecutionStatistics.modelName` on a round-trip that got an answer. A
     * round-trip that never answered has no such event.
     */
    data class StreamResponded(val servedModelName: String) : PromptExecutionEvent

    /**
     * A single content part the model just produced. The wrapped [MessagePart] tells the
     * consumer what flavour it is (text / thought / tool-call / inline-data / file).
     */
    data class PartReceived(val part: MessagePart) : PromptExecutionEvent

    /**
     * Token-usage delta. Both fields are increments since the previous delta — aggregator
     * sums them into the final [InputTokenStatistics] / [OutputTokenStatistics]. Providers
     * with cumulative `usage` (Gemini) subtract the previous snapshot before emitting.
     * Multiple deltas may be emitted per round-trip (one per stream chunk that updates
     * usage); aggregator just adds them up.
     */
    data class UsageDelta(
        val input: InputTokenStatistics = InputTokenStatistics.EMPTY,
        val output: OutputTokenStatistics = OutputTokenStatistics.EMPTY,
    ) : PromptExecutionEvent

    /**
     * Provider created a context cache as a separate API call alongside this round-trip.
     * Carries everything needed to write an `llm_statistics` row under
     * `nature = CACHE_CREATE`: which cache, how many tokens it covers, its TTL/scope, the
     * usage cost of the cache-create call, how long the create took, and the actual
     * content that went into the cache (for post-hoc analytics / debugging of "what
     * exactly was cached"). Side-channel event consumed by the agent — aggregator
     * ignores it.
     */
    data class CacheCreated(
        val cacheId: String,
        val modelName: String,
        val tokens: PromptExecutionTokenStatistics,
        val tokenCount: Long? = null,
        val ttl: Duration? = null,
        val scope: PromptCacheScope? = null,
        val duration: Duration,
        /**
         * What was placed in the cache, in provider-neutral form. Persisted to
         * `llm_statistics.raw_request_prompt` for the CACHE_CREATE row so admins can
         * see what each cache contains. Defaults to empty list for backwards-compat
         * with event constructors that don't fill it.
         */
        val requestMessages: List<PromptMessage> = emptyList(),
    ) : PromptExecutionEvent

    /**
     * Stream completed successfully. The Flow closes after this. `finishReason` is the
     * provider's raw stop reason (e.g. Gemini's `STOP` / `MAX_TOKENS`) — useful to keep on
     * the success path for analytics. `streamDiagnostics` carries per-attempt chunk/parts
     * metrics for streaming executors (null for direct/non-streaming).
     */
    data class Completed(
        val finishReason: String? = null,
        val streamDiagnostics: StreamDiagnostics? = null,
    ) : PromptExecutionEvent

    /**
     * Stream terminated with an error. Listeners get the event before the Flow throws.
     * `finishReason` is carried for failures that do come back with one (notably
     * `PolicyViolationException`, where the provider's `finish_reason` token like
     * `IMAGE_PROHIBITED_CONTENT` is exactly the categorisation we want downstream).
     * `streamDiagnostics` — what managed to arrive before the break: 0 chunks = connect failure,
     * many chunks + big `sinceLastChunkMs` = server-side cap, etc.
     */
    data class Failed(
        val error: Throwable,
        val finishReason: String? = null,
        val streamDiagnostics: StreamDiagnostics? = null,
    ) : PromptExecutionEvent
}
