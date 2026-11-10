package org.nobirds.murmuration.gemini

import com.google.genai.Client
import com.google.genai.ResponseStream
import com.google.genai.types.Content
import com.google.genai.types.EmbedContentConfig
import com.google.genai.types.FunctionCallingConfig
import com.google.genai.types.FunctionCallingConfigMode
import com.google.genai.types.GenerateContentConfig
import com.google.genai.types.GenerateContentResponse
import com.google.genai.types.HttpOptions
import com.google.genai.types.ImageConfig
import com.google.genai.types.Schema
import com.google.genai.types.ThinkingConfig
import com.google.genai.types.ThinkingLevel as GeminiThinkingLevel
import com.google.genai.types.ToolConfig
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.emitAll
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.future.await
import kotlinx.coroutines.launch
import kotlinx.coroutines.runInterruptible
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.serializer
import org.nobirds.murmuration.gemini.internal.logger
import org.nobirds.murmuration.prompt.ImageAspectRatio
import org.nobirds.murmuration.prompt.ImagePromptSettings
import org.nobirds.murmuration.prompt.ImageResolution
import org.nobirds.murmuration.prompt.MusicPromptSettings
import org.nobirds.murmuration.prompt.InstrumentDeclaration
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptCachingProvider
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.EmbeddingPromptSettings
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.toMessage
import org.nobirds.murmuration.prompt.OutputTokenStatistics
import org.nobirds.murmuration.prompt.InputTokenStatistics
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.StreamTimingTracker
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Failed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.UsageDelta
import org.nobirds.murmuration.prompt.PromptExecutionException
import org.nobirds.murmuration.prompt.finishReasonOrNull
import org.nobirds.murmuration.prompt.minus
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.PromptMessage
import org.nobirds.murmuration.prompt.MalformedResponseException
import org.nobirds.murmuration.prompt.ParsePhase
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.PromptSettings
import org.nobirds.murmuration.prompt.ThinkingLevel
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.ToolsPromptSettings
import org.nobirds.murmuration.prompt.TypedPromptSettings
import org.nobirds.murmuration.prompt.dump
import org.nobirds.murmuration.prompt.dumpSettings
import org.nobirds.murmuration.schema.toJsonSchema
import kotlin.jvm.optionals.getOrNull
import kotlin.reflect.KClass
import kotlin.reflect.KType
import java.util.concurrent.CompletableFuture
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.time.Duration
import kotlin.time.Duration.Companion.nanoseconds
import kotlin.time.Duration.Companion.seconds

@OptIn(ExperimentalSerializationApi::class)
class GeminiJsonSchemaProvider {
    fun generateSchemaFromType(type: KClass<*>): Schema {
        val serializer = serializer(type.java)
        return serializer.toJsonSchema().toGeminiSchema()
    }

    fun generateSchemaFromKType(type: KType): Schema {
        val serializer = serializer(type)
        return serializer.toJsonSchema().toGeminiSchema()
    }
}

/**
 * Strategy for obtaining a response from Gemini: a single request (Direct) or streaming chunks (Streaming).
 * Emits `PromptExecutionEvent` deltas as Parts arrive from the API; the final
 * `StatisticsUpdate` + `Completed` closes the stream. On any error `Failed` is emitted first,
 * then the exception is propagated further through the Flow.
 *
 * `configureRequest` — a hook for strategy-specific config tweaks (reserved for future
 * settings, does nothing by default).
 */
interface GeminiResponseStrategy {
    fun configureRequest(builder: GenerateContentConfig.Builder) {}

    fun execute(
        prompt: Prompt,
        modelName: String,
        contents: List<Content>,
        config: GenerateContentConfig,
        cacheResult: PromptCachingProvider.CacheResult?,
        requestMessages: List<PromptMessage>,
        requestedModelName: String,
    ): Flow<PromptExecutionEvent>
}

/**
 * Direct variant: a single `generateContent`; on the response we parse all Parts and emit them
 * as the corresponding events in one pass. Essentially this is streaming semantics in which
 * all events arrive synchronously after the single response returns.
 *
 * A single [generateContent] `await` has no timeout of its own — only OkHttp's socket
 * timeouts, which sit at ~10 min and, worse, don't fire while the server keeps the
 * connection open but silent. [overallTimeout] is a hard ceiling on the whole round-trip:
 * a request that hasn't returned within it is aborted and surfaced as a retryable
 * [TransientException] (NETWORK_TIMEOUT), the same signal the streaming path raises — so a
 * hung Direct call retries instead of blocking a pipeline for ~10 min.
 *
 * `withTimeout` alone is not enough: it cancels the coroutine but does NOT abort the underlying
 * OkHttp `Call`. In google-genai 1.56.0, `HttpApiClient.asyncExecuteRequest` enqueues the call
 * without ever storing it, and the returned `CompletableFuture` is not wired to `Call.cancel()`,
 * so cancelling the coroutine leaves a live, still-billable request behind — and a retry then runs
 * a second request in parallel with that zombie. To actually tear the socket down we attach a
 * per-request OkHttp `callTimeout` via [GenerateContentConfig.withCallTimeout]: the SDK rebuilds
 * the client with that `callTimeout` for THIS call only (it reads the request's `HttpOptions` tag).
 * The `callTimeout` is set to [overallTimeout] plus [callTimeoutMargin] so it stays a pure backstop:
 * `withTimeout` fires first and yields the clean [TransientException] mapping, and `callTimeout`
 * only bites (killing the socket) if `withTimeout` somehow doesn't.
 *
 * ⚠️ This `callTimeout` backstop is Direct-ONLY and MUST NOT be applied to the streaming path. A
 * `callTimeout` caps the whole call including reading the response body; on a stream the body is
 * read for the entire stream's lifetime, so a `callTimeout` would kill healthy long streams (music
 * with a 300s per-prompt idle in particular). [StreamingResponseStrategy] deliberately never sets it.
 */
class DirectResponseStrategy(
    private val client: Client,
    private val overallTimeout: Duration = 600.seconds,
    private val callTimeoutMargin: Duration = 30.seconds,
) : GeminiResponseStrategy {
    override fun execute(
        prompt: Prompt,
        modelName: String,
        contents: List<Content>,
        config: GenerateContentConfig,
        cacheResult: PromptCachingProvider.CacheResult?,
        requestMessages: List<PromptMessage>,
        requestedModelName: String,
    ): Flow<PromptExecutionEvent> = flow {
        val timedConfig = config.withCallTimeout(overallTimeout + callTimeoutMargin)
        // Before the request goes out: a round-trip that dies without an answer keeps its request/cache data.
        emit(startedEvent(requestMessages, cacheResult, requestedModelName))
        try {
            val response = withTimeout(overallTimeout) {
                client.async.models.generateContent(modelName, contents, timedConfig).await()
            }
            emit(response.respondedEvent(requestedModelName))
            val parts = response.allParts()
            for (part in parts) {
                part.toEventOrNull()?.let { emit(it) }
            }
            val input = response.readInputUsage()
            val output = response.toOutputUsage()
            if (input != InputTokenStatistics.EMPTY || output != OutputTokenStatistics.EMPTY)
                emit(UsageDelta(input = input, output = output))
            // Non-STOP finish reasons (MAX_TOKENS, SAFETY, MALFORMED_FUNCTION_CALL, RECITATION,
            // SPII, …) must surface as exceptions so the pipeline runs recovery instead of
            // accepting a truncated/policy-blocked response as success. Without this,
            // a MAX_TOKENS reply from a looping agent silently aggregates thousands of
            // tool-calls into pipeline state and we only learn about it post-mortem.
            val finishReason = response.finishReasonString()
            mapFinishReason(finishReason, prompt)?.let { mapped ->
                emit(Failed(mapped, finishReason = finishReason))
                throw mapped
            }
            emit(Completed(finishReason))
        } catch (e: TimeoutCancellationException) {
            // Overall round-trip ceiling hit (see [overallTimeout]). withTimeout raises this only
            // on its own deadline, never on genuine external cancellation, so it's safe to map to a
            // retryable TransientException rather than propagate as a cancel. Caught before the
            // CancellationException branch below because TimeoutCancellationException is one.
            val mapped = TransientException(
                TransientReason.NETWORK_TIMEOUT,
                "Gemini direct request exceeded overall round-trip budget of ${overallTimeout.inWholeSeconds}s",
                e,
            )
            emit(Failed(mapped, finishReason = null))
            throw mapped
        } catch (e: CancellationException) {
            throw e
        } catch (e: PromptExecutionException) {
            // Already mapped (e.g. from mapFinishReason above) — re-throw without remapping.
            throw e
        } catch (e: Throwable) {
            val mapped = e.toPromptExecutionException(prompt)
            emit(Failed(mapped, finishReason = mapped.finishReasonOrNull()))
            throw mapped
        }
    }
}

/**
 * Returns a copy of [this] config carrying a per-request OkHttp `callTimeout` (encoded as
 * `HttpOptions.timeout`, in milliseconds), preserving any [HttpOptions] already set on the config.
 *
 * This is the only reliable way to abort a hung google-genai call: the SDK never exposes the
 * underlying OkHttp `Call`, so a coroutine-side `withTimeout` cannot tear the socket down. When the
 * per-request `HttpOptions` tag carries a timeout, `HttpApiClient.asyncExecuteRequest` rebuilds the
 * OkHttp client with a matching `callTimeout` for that single call, which really kills the socket.
 *
 * ⚠️ Direct path ONLY. A `callTimeout` bounds the whole call including response-body reads; a
 * streaming response reads its body for the stream's entire lifetime, so a `callTimeout` there would
 * abort healthy long streams. See [DirectResponseStrategy] / [StreamingResponseStrategy].
 */
internal fun GenerateContentConfig.withCallTimeout(timeout: Duration): GenerateContentConfig {
    val timeoutMillis = timeout.inWholeMilliseconds.coerceIn(1L, Int.MAX_VALUE.toLong()).toInt()
    val httpOptions = httpOptions().orElse(HttpOptions.builder().build())
        .toBuilder()
        .timeout(timeoutMillis)
        .build()
    return toBuilder().httpOptions(httpOptions).build()
}

/**
 * Streaming variant: `generateContentStream` returns a `ResponseStream`, which we drain
 * as a `Flow` and emit events as Parts arrive in each chunk. In parallel we
 * accumulate all Parts — the final `StatisticsUpdate` is built from the metadata of the last
 * chunk (where Gemini sets the final `usageMetadata`/`finishReason`) and the accumulated
 * Parts (to assemble `responseMessage`).
 *
 * Per-chunk idle timeout: silence longer than [idleTimeout] turns into `InterruptedIOException`,
 * which the retry mechanism picks up. Thought logging goes through `.onEach` at the tail of the Flow,
 * without a special branch in the strategy.
 *
 * On top of the per-chunk idle timeout, [overallTimeout] caps the **whole** stream. The idle
 * timeout alone is insufficient: every chunk resets it, so a request that keeps trickling chunks
 * (e.g. long thinking) can drag on far past any sane budget without ever tripping idle. The
 * overall cap is checked before each chunk pull and, once exceeded, raises the same retryable
 * [TransientException] (NETWORK_TIMEOUT) as idle. It is a hard ceiling, not a per-chunk bound, so
 * a legitimate long stream stays fine as long as its total wall time stays under [overallTimeout].
 *
 * Every timeout (first event, inter-chunk idle, overall) and every other exit tears the underlying
 * OkHttp `Call` down via [callTracker] BEFORE the failure propagates, so a retry never runs alongside
 * a zombie request. [client] MUST have been built with `.clientOptions(callTracker.clientOptions())`;
 * the first request fails loudly otherwise.
 */
class StreamingResponseStrategy(
    private val client: Client,
    private val callTracker: StreamCallTracker,
    private val idleTimeout: Duration,
    private val overallTimeout: Duration = 600.seconds,
) : GeminiResponseStrategy {

    private val logger = logger()

    override fun execute(
        prompt: Prompt,
        modelName: String,
        contents: List<Content>,
        config: GenerateContentConfig,
        cacheResult: PromptCachingProvider.CacheResult?,
        requestMessages: List<PromptMessage>,
        requestedModelName: String,
    ): Flow<PromptExecutionEvent> = flow {
        // Per-prompt override via PromptSettings; null → global default from the constructor.
        val effectiveIdleTimeout = prompt.settings?.streamIdleTimeout ?: idleTimeout
        var stream: ResponseStream<GenerateContentResponse>? = null
        // Stream diagnostics tracker — filled as chunks arrive. On failure
        // we pull it into the log and the exception message to see "after how many chunks
        // it failed, how long we waited for the last one, whether there was a finishReason". Without this the SDK hides
        // the detail in an IOException → the break is indistinguishable from normal completion in the logs.
        val streamStartedAt = System.nanoTime()
        var chunkCount = 0
        var lastChunkAt = streamStartedAt
        var lastFinishReason: String? = null
        var totalTextParts = 0
        var totalThoughtParts = 0
        var totalToolCallParts = 0
        // time-to-first-chunk / max inter-chunk gap — tracked unconditionally
        // (not gated by isDebugEnabled, unlike `elapsedMs` below), so they exist on every
        // StreamDiagnostics, including with DEBUG off.
        val timing = StreamTimingTracker(streamStartedAt)
        // Handle on this request's OkHttp Call (filled by the tracker's interceptor when the request
        // starts). The ticket is opened INSIDE the try, right before the request and AFTER the
        // suspending `emit`: a collector cancelled on `PromptStarted` leaves before any ticket
        // exists (an abort before the call registers would keep the ticket by design), and every
        // later exit reaches the `finally` that cancels the Call.
        var openedCall: StreamCall? = null
        try {
            // Before the request goes out: a stream that dies before its first chunk (TTFB timeout)
            // keeps its request/cache data in the round-trip's statistics.
            emit(startedEvent(requestMessages, cacheResult, requestedModelName))
            val streamCall = callTracker.open().also { openedCall = it }
            // TTFB guard: the SDK reads the first SSE event inside this future, so the whole
            // time-to-first-byte sits here — before the per-chunk idle loop / overall deadline start.
            // Bound it by the idle timeout (silence before the first event is idle); on timeout the
            // Call is cancelled before the retryable exception is thrown. See [awaitFirstEventOrTimeout].
            stream = awaitFirstEventOrTimeout(
                idleTimeout = effectiveIdleTimeout,
                onTimeout = streamCall::abort,
                future = client.async.models.generateContentStream(modelName, contents, streamCall.applyTo(config)),
            )
            check(streamCall.isRegistered) {
                "StreamCallTracker interceptor never saw the request: the genai Client was not built " +
                    "with StreamCallTracker.clientOptions(), so a timed-out stream could not be aborted"
            }
            var lastChunk: GenerateContentResponse? = null
            var respondedEmitted = false
            var previousInput = InputTokenStatistics.EMPTY
            var previousOutput = OutputTokenStatistics.EMPTY

            stream.asChunkFlow(effectiveIdleTimeout, overallTimeout, abortPull = streamCall::abort).collect { chunk ->
                chunkCount++
                lastChunkAt = System.nanoTime()
                timing.recordChunk(lastChunkAt)
                lastChunk = chunk
                val parts = chunk.allParts()
                if (!respondedEmitted) {
                    emit(chunk.respondedEvent(requestedModelName))
                    respondedEmitted = true
                }
                // Counting parts-of-type for diagnostics (without a full content dump — that
                // goes into raw_response_text).
                for (part in parts) {
                    when {
                        part.functionCall().isPresent -> totalToolCallParts++
                        part.thought().orElse(false) -> totalThoughtParts++
                        part.text().isPresent -> totalTextParts++
                    }
                }
                // finishReason of the last chunk — if non-null, the model finished normally.
                val fr = chunk.candidates().orElse(emptyList()).firstOrNull()
                    ?.finishReason()?.orElse(null)
                if (fr != null) lastFinishReason = fr.toString()
                for (part in parts) {
                    part.toEventOrNull()?.let { emit(it) }
                }
                // Gemini sends usage as cumulative (a growing snapshot) — we turn it into deltas
                // by subtracting the previous snapshot. The aggregator then sums the deltas back.
                // Input: `cachedContentTokenCount` is known from the first chunk, while
                // `promptTokenCount` (the non-cached part) usually appears only in the last —
                // hence it is **cumulative**, not "once at the start".
                val cumulativeInput = chunk.readInputUsage()
                val cumulativeOutput = chunk.toOutputUsage()
                val inputDelta = cumulativeInput - previousInput
                val outputDelta = cumulativeOutput - previousOutput
                if (inputDelta != InputTokenStatistics.EMPTY || outputDelta != OutputTokenStatistics.EMPTY) {
                    emit(UsageDelta(input = inputDelta, output = outputDelta))
                    previousInput = cumulativeInput
                    previousOutput = cumulativeOutput
                }
                if (logger.isDebugEnabled) {
                    val elapsedMs = (lastChunkAt - streamStartedAt) / 1_000_000
                    logger.debug(
                        "Gemini stream chunk #{}: model={}, parts={} (text={}, thought={}, tool_call={}), finish_reason={}, elapsed={}ms",
                        chunkCount, modelName, parts.size, totalTextParts, totalThoughtParts, totalToolCallParts,
                        lastFinishReason, elapsedMs,
                    )
                }
            }

            lastChunk ?: throw TransientException(
                TransientReason.EMPTY_STREAM,
                "Gemini streaming returned 0 chunks before stream end",
                null,
            )
            // Non-STOP finish reasons (MAX_TOKENS, SAFETY, MALFORMED_FUNCTION_CALL, RECITATION,
            // SPII, …) must surface as exceptions so the pipeline runs recovery instead of
            // accepting a truncated/policy-blocked response as success. Without this,
            // a MAX_TOKENS reply from a looping agent silently aggregates thousands of
            // tool-calls into pipeline state and we only learn about it post-mortem.
            val diag = timing.snapshot(chunkCount, totalTextParts, totalThoughtParts, totalToolCallParts, lastFinishReason)
            mapFinishReason(lastFinishReason, prompt)?.let { mapped ->
                mapped.streamDiagnostics = diag
                emit(Failed(mapped, finishReason = lastFinishReason, streamDiagnostics = diag))
                throw mapped
            }
            emit(Completed(lastFinishReason, streamDiagnostics = diag))
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            val diag = timing.snapshot(chunkCount, totalTextParts, totalThoughtParts, totalToolCallParts, lastFinishReason)
            logger.warn(
                "Gemini stream failed [{}]: {} — model={}, chunks={}, total_elapsed={}ms, since_last_chunk={}ms, last_finish_reason={}, parts_text={}, parts_thought={}, parts_tool_call={}",
                e::class.simpleName, e.message, modelName, diag.chunkCount, diag.totalElapsedMs,
                diag.sinceLastChunkMs, diag.lastChunkFinishReason, diag.partsText, diag.partsThought, diag.partsToolCall,
            )
            val mapped = e.toPromptExecutionException(prompt)
            mapped.streamDiagnostics = diag
            emit(Failed(mapped, finishReason = mapped.finishReasonOrNull(), streamDiagnostics = diag))
            throw mapped
        } finally {
            // Ignore close failures — OkHttp may throw if the socket was already closed
            // by the remote side; there is nothing left to recover here.
            runCatching { stream?.close() }
            // Cancel the Call on EVERY exit (idle/overall timeout, failure, collector cancellation,
            // normal end — a no-op for a finished call) so no request outlives its attempt.
            openedCall?.abort()
        }
    }.onEach { event ->
        // Thoughts — TRACE: they are already written to `llm_statistics.raw_response_text` as
        // separate MessagePart.Thought, available through analysis from the DB.
        if (event is PromptExecutionEvent.PartReceived && logger.isTraceEnabled) {
            val thought = event.part as? MessagePart.Thought ?: return@onEach
            logger.trace("***Thoughts*** {}", thought.text.replace("\n", " "))
        }
    }
}

/**
 * Awaits the streaming request's first SSE event under an [idleTimeout], mapping a silent server to
 * a retryable [TransientException] (NETWORK_TIMEOUT).
 *
 * The genai SDK eagerly reads the FIRST SSE event inside the future returned by
 * `generateContentStream` (`ResponseStream` blocks on the first line at construction), so the whole
 * time-to-first-byte sits inside this `await`, BEFORE the per-chunk idle loop or the overall
 * deadline ever start ticking. Left unguarded, a server that accepts the connection but never sends
 * the first event would hang here indefinitely (subject only to OkHttp's ~10-min socket timeout).
 * Silence before the first event is semantically idle, so we bound it by the same [idleTimeout] the
 * per-chunk loop uses.
 *
 * On timeout [onTimeout] runs FIRST — it must cancel the underlying OkHttp `Call` (a `callTimeout`
 * cannot be used on a streaming request, it would kill healthy long streams, see [withCallTimeout];
 * the SDK exposes no `Call` handle, see [StreamCallTracker]) — and only then the retryable
 * exception is thrown. So by the time any retry starts, the timed-out request is already cancelled.
 */
internal suspend fun <T> awaitFirstEventOrTimeout(
    idleTimeout: Duration,
    onTimeout: () -> Unit,
    future: CompletableFuture<T>,
): T = try {
    withTimeout(idleTimeout) { future.await() }
} catch (e: TimeoutCancellationException) {
    onTimeout()
    throw TransientException(
        TransientReason.NETWORK_TIMEOUT,
        "Gemini streaming produced no first event within ${idleTimeout.inWholeSeconds}s " +
            "(TTFB idle; request cancelled)",
        e,
    )
}

/**
 * Drains the Gemini `ResponseStream` as `Flow<GenerateContentResponse>`. Each chunk
 * is read on `Dispatchers.IO` via `runInterruptible`. Between chunks — an idle timeout;
 * silence longer than [idleTimeout] turns into `InterruptedIOException`.
 *
 * On top of idle, [overallTimeout] is a hard ceiling on the total streaming wall time: because
 * every chunk resets idle, a stream that keeps trickling chunks would otherwise never trip idle.
 * The deadline is checked before each chunk pull; exceeding it raises a retryable
 * [TransientException] (NETWORK_TIMEOUT), the same class idle produces.
 */
internal fun ResponseStream<GenerateContentResponse>.asChunkFlow(
    idleTimeout: Duration,
    overallTimeout: Duration,
    abortPull: () -> Unit,
): Flow<GenerateContentResponse> = iterator().asTimedChunkFlow(idleTimeout, overallTimeout, abortPull)

/**
 * Drains an [Iterator] as a [Flow] under two timeouts: a per-element [idleTimeout] (silence
 * between elements) and a hard [overallTimeout] on total wall time. The overall deadline is
 * captured at collection start; before every pull the wait is bounded by the SMALLER of the
 * remaining budget and [idleTimeout], so the overall ceiling can no longer be overshot by up to a
 * full [idleTimeout] on a chunk that stalls right before the deadline (with music's 300s per-prompt
 * idle that overshoot nearly doubled the budget). Exceeding either limit raises a retryable
 * [TransientException] (NETWORK_TIMEOUT); the message names whichever limit bound the pull.
 * Extracted from [asChunkFlow] so the timeout behaviour is unit-testable with a plain fake iterator
 * (no genai `ResponseStream` needed).
 */
internal fun <T : Any> Iterator<T>.asTimedChunkFlow(
    idleTimeout: Duration,
    overallTimeout: Duration,
    abortPull: () -> Unit,
): Flow<T> = flow {
    val overallDeadlineAt = System.nanoTime() + overallTimeout.inWholeNanoseconds
    while (true) {
        val remaining = (overallDeadlineAt - System.nanoTime()).nanoseconds
        if (remaining <= Duration.ZERO) throw overallBudgetTimeout(overallTimeout)
        // Bind the pull by whichever limit is nearer. When the overall budget is the binding one,
        // a stalled chunk trips at the deadline instead of overshooting to a full idleTimeout past it.
        val boundByOverall = remaining < idleTimeout
        val next = try {
            pollNext(minOf(idleTimeout, remaining), abortPull)
        } catch (_: PullStalled) {
            throw if (boundByOverall) overallBudgetTimeout(overallTimeout)
            else TransientException(
                TransientReason.NETWORK_TIMEOUT,
                "Gemini streaming idle: no chunk received within ${idleTimeout.inWholeSeconds}s (timeout/connection lost)",
                null,
            )
        } ?: break
        emit(next)
    }
}

private fun overallBudgetTimeout(overallTimeout: Duration) = TransientException(
    TransientReason.NETWORK_TIMEOUT,
    "Gemini streaming exceeded overall round-trip budget of " +
        "${overallTimeout.inWholeSeconds}s (idle timeout kept resetting on chunks)",
    null,
)

/** The pull outlived its budget and was aborted by the watchdog in [pollNext]. */
private class PullStalled : RuntimeException()

/**
 * Pulls the next element on `Dispatchers.IO` under [timeout]. A plain `withTimeout` around
 * `runInterruptible` does NOT work for a silent socket: `runInterruptible` only interrupts the thread,
 * a blocking socket read ignores interrupts, and the coroutine then waits for the read to return —
 * so the idle timeout never fired (found by `StreamCallTrackerTest`: the test hung forever). The
 * watchdog therefore calls [abortPull] (cancels the OkHttp `Call`, which makes the blocked read throw)
 * and interrupts the pull; the resulting failure is reported as [PullStalled]. [abortPull] also runs if
 * the collector is cancelled mid-pull, so cancellation cannot be stuck behind a blocked read either.
 */
private suspend fun <T : Any> Iterator<T>.pollNext(timeout: Duration, abortPull: () -> Unit): T? = coroutineScope {
    val stalled = AtomicBoolean(false)
    // runCatching keeps a read failure inside the Result so it cannot cancel this scope before we classify it.
    val pull = async(Dispatchers.IO) {
        runCatching { runInterruptible { if (hasNext()) next() else null } }
    }
    val watchdog = launch {
        delay(timeout)
        stalled.set(true)
        abortPull()
        pull.cancel()
    }
    try {
        pull.await().getOrThrow()
    } catch (e: CancellationException) {
        if (stalled.get()) throw PullStalled()
        abortPull() // collector cancelled mid-pull: unblock the read so cancellation can complete
        throw e
    } catch (e: Throwable) {
        if (stalled.get()) throw PullStalled()
        throw e
    } finally {
        watchdog.cancel()
    }
}

/**
 * Launches a background cache keep-alive task in parallel with executing [block]: while
 * [block] runs, the provider regularly (every [interval]) checks the cache state and
 * decides on its own whether to extend its TTL. If the provider returned `null` (cache expired/deleted) —
 * the keeper stops, there is no point hitting the API further. When [block] finishes (normally
 * or with an exception) the keeper is cancelled.
 *
 * If [cacheId] or the provider itself is `null`, simply runs [block] without keep-alive.
 */
internal suspend fun <T> PromptCachingProvider?.withCacheKeepAlive(
    cacheId: String?,
    interval: Duration,
    block: suspend () -> T,
): T {
    if (cacheId == null || this == null) return block()
    return coroutineScope {
        val keeperJob = launch { runKeeperLoop(cacheId, interval) }
        try {
            block()
        } finally {
            keeperJob.cancel()
        }
    }
}

/**
 * Flow variant of keep-alive: the keeper lives exactly as long as collection of [this] lasts. While
 * there is a consumer the cache is extended; when the stream closes (Completed / Failed / cancellation) —
 * the keeper stops. Used to wrap the event stream from
 * [GeminiResponseStrategy].
 */
internal fun <T> Flow<T>.withCacheKeepAlive(
    provider: PromptCachingProvider?,
    cacheId: String?,
    interval: Duration,
): Flow<T> {
    if (cacheId == null || provider == null) return this
    val source = this
    return flow {
        coroutineScope {
            val keeperJob = launch { provider.runKeeperLoop(cacheId, interval) }
            try {
                source.collect { emit(it) }
            } finally {
                keeperJob.cancel()
            }
        }
    }
}

private suspend fun PromptCachingProvider.runKeeperLoop(cacheId: String, interval: Duration) {
    val keeperLogger = logger()
    while (true) {
        delay(interval) // throws CancellationException on cancel — exits the loop
        val stillAlive = try {
            keepAlive(cacheId)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            keeperLogger.warn("Cache keep-alive iteration failed for $cacheId: ${e.message}")
            continue
        }
        if (stillAlive == null) {
            keeperLogger.warn("Cache $cacheId is gone — stopping keep-alive loop")
            break
        }
    }
}

class GeminiClient(
    private val geminiClient: Client,
    private val wireKeyLayout: GeminiWireKeyLayout,
    private val cachingProvider: PromptCachingProvider? = null,
    private val schemaProvider: GeminiJsonSchemaProvider = GeminiJsonSchemaProvider(),
    private val responseStrategy: GeminiResponseStrategy = DirectResponseStrategy(geminiClient),
    private val cacheKeepAliveInterval: Duration = 60.seconds,
) : PromptExecutor {

    private val logger = logger()

    // A single path for text and images. Image generation is the same round-trip with
    // modality=IMAGE + imageConfig (details in executeText): the same strategy from config
    // (Direct/Streaming), the same cache and keep-alive, the same thinkingConfig and system-as-user.
    // Previously images went through a separate non-streaming call that lost thoughts and the system way.
    override fun execute(prompt: Prompt): Flow<PromptExecutionEvent> =
        when (prompt.settings) {
            is EmbeddingPromptSettings -> executeEmbedding(prompt)
            is MusicPromptSettings -> executeMusic(prompt)
            else -> executeText(prompt)
        }

    // Gemini-family executor: serves Vertex AI and the Gemini API (the genai SDK client is configured
    // for one of them at construction). Distinguishes this executor from other providers when
    // [RoutingPromptExecutor] selects by the prompt's model.
    override fun supports(llm: Llm): Boolean = llm.provider in setOf("vertex", "google")

    private fun executeText(prompt: Prompt): Flow<PromptExecutionEvent> = flow {
        val cacheResult = cachingProvider?.processPrompt(prompt) ?: PromptCachingProvider.CacheResult(prompt = prompt)
        val actualPrompt = cacheResult.prompt
        val cacheId = cacheResult.id

        // CACHE_CREATE is a separate row in `llm_statistics`. Emitted before the round-trip stream;
        // the agent reports it via `Flow.onEach` under its own display name. The aggregate
        // event ignores it — it is a side channel.
        cacheResult.createStatistics?.let { createStats ->
            emit(
                PromptExecutionEvent.CacheCreated(
                    cacheId = createStats.cacheId ?: cacheId ?: "",
                    modelName = createStats.modelName,
                    tokens = createStats.tokens,
                    tokenCount = createStats.cachedTokenCount,
                    ttl = createStats.cacheTtlMillis?.let { kotlin.time.Duration.parse("${it}ms") },
                    scope = createStats.cacheScope,
                    duration = cacheResult.createDuration ?: kotlin.time.Duration.ZERO,
                    requestMessages = createStats.requestMessages,
                )
            )
        }

        // SYSTEM content → a real Gemini systemInstruction (Content without a role). The model follows
        // system instructions noticeably better than the same rules smeared across the user message (e.g.
        // "no parentheses in names"). Symmetric to the cache path: GeminiCacheBackend.createCache also
        // puts SYSTEM into the cache's systemInstruction — previously the uncached path demoted it to user.
        // Tool declarations and history come from one build, so they share one wire key layout.
        val wireDeclarations = GeminiToolDeclarations.build(
            (actualPrompt.settings as? ToolsPromptSettings)?.tools?.flatMap { it.instruments }.orEmpty(),
            wireKeyLayout,
        )
        val converter = wireDeclarations.historyConverter
        val systemInstruction = converter.toSystemInstruction(
            actualPrompt.content.filter { it.type == PromptContentType.SYSTEM }.flatMap { it.parts }
        )

        val allContents = converter.toGenaiContents(
            actualPrompt.content.filter { it.type != PromptContentType.SYSTEM }
        )

        val configBuilder = GenerateContentConfig.builder()
        if (cacheId != null) configBuilder.cachedContent(cacheId)
        // systemInstruction only without a cache: with a cache it is already baked into the cached content, and Vertex
        // forbids re-sending system_instruction in a request that uses cached content.
        when {
            systemInstruction == null -> {} // no system, or all of it went into the cache — ok
            cacheId == null -> configBuilder.systemInstruction(systemInstruction)
            // cacheId != null, but SYSTEM parts remain in actualPrompt — this means a NO_CACHE system
            // next to the cache. It was not baked into the cache (marked non-cacheable), and Vertex does not allow
            // sending system_instruction together with cached content. The model will NOT get these rules in this call —
            // failing the pipeline over this is heavy-handed, but staying silent is also wrong (it would surface as murky artifacts).
            // A loud error in the log with diagnostics — so it is visible and clear how to fix.
            else -> logger.error(
                "Non-cacheable SYSTEM content present alongside cached content (cacheId={}). " +
                    "Vertex forbids system_instruction on cached requests, and these parts were not " +
                    "baked into the cache — they will NOT reach the model this call. Mark all SYSTEM " +
                    "parts as cacheable (PIPELINE) so they go into the cache, or move volatile system " +
                    "text to the user role.",
                cacheId,
            )
        }
        responseStrategy.configureRequest(configBuilder)
        applyPromptSettings(configBuilder, actualPrompt.settings, cacheId, actualPrompt.llm, wireDeclarations)

        // Image generation = a regular round-trip with modality=IMAGE + imageConfig. Strategy (stream/
        // direct), cache, thinkingConfig (applyPromptSettings already took includeThoughts/thinkingLevel
        // from ImagePromptSettings) and systemInstruction are shared with text. The image comes back as an
        // InlineData part → the BinaryDataStorage wrapper puts it into a file, thoughts settle into responseMessage.
        (actualPrompt.settings as? ImagePromptSettings)?.let { imageSettings ->
            // TEXT+IMAGE: under IMAGE-only the model thinks (thinking_tokens are billed) but does NOT
            // return thought parts — thoughts cannot be captured. With TEXT in the modality the model returns thoughts (and text)
            // as parts alongside the image; the image still arrives as InlineData → file.
            configBuilder.responseModalities(listOf("TEXT", "IMAGE"))
            configBuilder.imageConfig(
                ImageConfig.builder()
                    .imageSize(imageSettings.size.resolution.toGemini())
                    .aspectRatio(imageSettings.size.aspectRatio.toGemini())
                    .build()
            )
        }


        // Full dump of the prompt — TRACE: the content already sits in `llm_statistics.raw_request_prompt`,
        // duplicating it in regular logs makes no sense. INFO/DEBUG show the reader
        // structural events, the content is fetched from the DB.
        if (logger.isTraceEnabled) {
            logger.trace(
                "Sending Gemini request [cacheId={}, settings={}]\n{}",
                cacheId,
                actualPrompt.settings.dumpSettings(),
                actualPrompt.dump()
            )
        }

        val requestMessages = allContents.toMessages()
        val downstream = responseStrategy.execute(
            prompt = actualPrompt,
            modelName = prompt.llm.model,
            contents = allContents,
            config = configBuilder.build(),
            cacheResult = cacheResult,
            requestMessages = requestMessages,
            requestedModelName = prompt.llm.model,
        ).withCacheKeepAlive(cachingProvider, cacheId, cacheKeepAliveInterval)

        emitAll(downstream)
    }

    // Embedding (embedContent) is a different API method, so the branching is above the generateContent level.
    // Emits the SAME events (PromptStarted/UsageDelta/PartReceived/Completed) → statistics, cost and
    // listeners are shared through `aggregate()`, as for text/images.
    private fun executeEmbedding(prompt: Prompt): Flow<PromptExecutionEvent> = flow {
        val settings = prompt.settings as EmbeddingPromptSettings
        val text = prompt.content
            .flatMap { it.parts }
            .filterIsInstance<TextPromptPart>()
            .joinToString("\n") { it.text }
        val requestMessages = prompt.content.map { it.toMessage() }

        emit(
            PromptExecutionEvent.PromptStarted(
                modelName = prompt.llm.model,
                requestedModelName = prompt.llm.model,
                requestMessages = requestMessages,
            )
        )

        try {
            val config = EmbedContentConfig.builder()
                .taskType(settings.taskType)
                .outputDimensionality(settings.outputDimensionality)
                .build()
            val response = geminiClient.async.models.embedContent(prompt.llm.model, text, config).await()
            val values = response.embeddings().orElse(null)?.firstOrNull()
                ?.values()?.orElse(null)?.toFloatArray()
                ?: error("embedContent returned no embedding values for model '${prompt.llm.model}'")

            // Vertex bills embeddings by characters (metadata.billableCharacterCount); we put it into input
            // as a usage signal for cost reporting (precision is not critical — the spend is pennies).
            // Event order contract: PromptStarted -> StreamResponded -> everything else.
            emit(PromptExecutionEvent.StreamResponded(servedModelName = prompt.llm.model))
            response.metadata().orElse(null)?.billableCharacterCount()?.orElse(null)?.let { chars ->
                emit(PromptExecutionEvent.UsageDelta(input = InputTokenStatistics(total = chars.toLong(), cached = 0, cacheWriteShort = 0, cacheWriteLong = 0)))
            }
            emit(PromptExecutionEvent.PartReceived(MessagePart.Embedding(values.size, values)))
            emit(PromptExecutionEvent.Completed())
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            val mapped = e.toPromptExecutionException(prompt)
            emit(PromptExecutionEvent.Failed(mapped))
            throw mapped
        }
    }

    // Music (Lyria 3) generation: a clean, NON-streaming generateContent requesting AUDIO modality
    // — matching Google's Lyria sample. Deliberately bypasses executeText's text-model config
    // (thinking, maxOutputTokens, system-instruction, streaming), none of which Lyria accepts —
    // sending any of them yields "400 invalid argument". Audio returns as an InlineData(audio/mpeg)
    // part → the BinaryDataStorage wrapper persists it to a file (same as image output).
    private fun executeMusic(prompt: Prompt): Flow<PromptExecutionEvent> = flow {
        // No tools in a music request: the converter has no wire key maps.
        val contents = GeminiHistoryConverter.EMPTY.toGenaiContents(
            prompt.content.filter { it.type != PromptContentType.SYSTEM }
        )
        // Clean config: only the AUDIO modality. No thinking / maxOutputTokens / system-instruction —
        // Lyria rejects those with "400 invalid argument". Streaming-vs-direct is left to the
        // configured [responseStrategy] (same as text/image), not hardcoded.
        val config = GenerateContentConfig.builder()
            .responseModalities(listOf("AUDIO", "TEXT"))
            .build()
        emitAll(
            responseStrategy.execute(
                prompt = prompt,
                modelName = prompt.llm.model,
                contents = contents,
                config = config,
                cacheResult = null,
                requestMessages = contents.toMessages(),
                requestedModelName = prompt.llm.model,
            )
        )
    }

    private fun applyPromptSettings(
        configBuilder: GenerateContentConfig.Builder,
        settings: PromptSettings?,
        cacheId: String?,
        llm: Llm,
        wireDeclarations: WireDeclarations,
    ) {
        // Thinking return — a single point for streaming/non-streaming. Billing of thinking tokens
        // does not depend on the `includeThoughts` flag (thoughts are always computed on the model side),
        // the flag only controls whether the trace is returned outward.
        //
        // Two levers: `thinkingBudget` (a numeric cap for Flash 2.5/3, 0=off / -1=auto / N=tokens)
        // and `thinkingLevel` (an enum tier for Pro-class models, where the API does not accept a number).
        // We detect Pro-class by model name — it cannot be guaranteed via a provider-neutral
        // attribute without extending [Llm], and the name is stable by Google convention.
        //
        // We pass exactly what the caller specified. Compatibility with a particular model is
        // the caller's responsibility (via `pipelines.turn.singleAgent*` and the like).
        // If the model does not understand `thinking_level` the provider returns 400; the caller will see it
        // and fix the config. No "name contains pro" heuristics.
        val tcb = ThinkingConfig.builder().includeThoughts(settings?.includeThoughts ?: true)
        settings?.thinkingLevel?.let { tcb.thinkingLevel(it.toGemini()) }
        settings?.thinkingBudget?.let { tcb.thinkingBudget(it) }
        configBuilder.thinkingConfig(tcb.build())
        when (settings) {
            null -> {}
            is TypedPromptSettings<*> -> {
                configBuilder.responseMimeType("application/json")
                configBuilder.responseSchema(schemaProvider.generateSchemaFromType(settings.type))
            }
            is ToolsPromptSettings -> {
                // Vertex AI: when using cached content it is **forbidden** to set tools /
                // tool_config / system_instruction in the request — Gemini returns 400
                // "Tool config, tools and system instruction should not be set in the
                // request when using cached content". The allowed list is fixed at the moment
                // the cache is created; per-phase filtering at the provider level for cacheable
                // agents is impossible. Filtering goes through a runtime guard in ToolCallingAgent
                // (line ~541) + prompt instructions about the tools available in the current phase.
                if (cacheId != null) return
                val tool = wireDeclarations.tool ?: return
                configBuilder.tools(listOf(tool))
                // Always AUTO. Provider-level allowlisting via mode=ANY+allowedFunctionNames is
                // incompatible with cached content (Vertex returns 400 when tool_config is set
                // on requests reusing a cache), so per-phase tool gating is enforced runtime-side
                // by [ToolCallingAgent]'s ToolsExecutor instead.
                val functionCallingConfig = FunctionCallingConfig.builder()
                    .mode(FunctionCallingConfigMode("AUTO"))
                    .build()
                configBuilder.toolConfig(
                    ToolConfig.builder()
                        .functionCallingConfig(functionCallingConfig)
                        .build()
                )
            }
            else -> {}
        }
        // Output-token cap. Explicit per-prompt; null = let the provider apply its own
        // default (Gemini Flash ≈ 64K). The cap covers content + thinking tokens combined,
        // so picking it requires knowing the model's thinking-budget regime — that's a
        // caller-side concern, set via [withMaxOutputTokens] on the prompt at the pipeline
        // wiring layer, not a model constant.
        // Hitting the cap surfaces as `finish_reason=MAX_TOKENS` → [MaxTokensException],
        // so a runaway loop fails fast instead of silently aggregating thousands of
        // tool-calls into pipeline state.
        settings?.maxOutputTokens?.let { configBuilder.maxOutputTokens(it) }
    }
}

// Shared Extension Functions

internal fun ThinkingLevel.toGemini(): GeminiThinkingLevel = when (this) {
    ThinkingLevel.MINIMAL -> GeminiThinkingLevel(GeminiThinkingLevel.Known.MINIMAL)
    ThinkingLevel.LOW -> GeminiThinkingLevel(GeminiThinkingLevel.Known.LOW)
    ThinkingLevel.MEDIUM -> GeminiThinkingLevel(GeminiThinkingLevel.Known.MEDIUM)
    ThinkingLevel.HIGH -> GeminiThinkingLevel(GeminiThinkingLevel.Known.HIGH)
}

internal fun ImageResolution.toGemini(): String = when (this) {
    // The Gemini image API does not support 512; clamp to 1K so as not to fabricate an invalid
    // imageSize and not crash the pipeline with a 400 error (the lite model accepts only 1K).
    ImageResolution.SD_512 -> "1K"
    ImageResolution.HD_1024 -> "1K"
    ImageResolution.XHD_2048 -> "2K"
    ImageResolution.UHD_4096 -> "4K"
}

private fun ImageAspectRatio.toGemini(): String = when (this) {
    ImageAspectRatio.SQUARE_1_1 -> "1:1"

    ImageAspectRatio.PORTRAIT_9_16 -> "9:16"
    ImageAspectRatio.PORTRAIT_4_3 -> "4:3"
    ImageAspectRatio.PORTRAIT_16_9 -> "16:9"
    ImageAspectRatio.PORTRAIT_8_1 -> "8:1"
    ImageAspectRatio.PORTRAIT_3_2 -> "3:2"
    ImageAspectRatio.PORTRAIT_4_5 -> "4:5"
    ImageAspectRatio.PORTRAIT_21_9 -> "21:9"

    ImageAspectRatio.LANDSCAPE_16_9 -> "16:9"
    ImageAspectRatio.LANDSCAPE_3_4 -> "3:4"
    ImageAspectRatio.LANDSCAPE_9_16 -> "9:16"
    ImageAspectRatio.LANDSCAPE_1_8 -> "1:8"
    ImageAspectRatio.LANDSCAPE_2_3 -> "2:3"
    ImageAspectRatio.LANDSCAPE_5_4 -> "5:4"
}

