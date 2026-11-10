package org.nobirds.murmuration.prompt

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlin.time.Duration

/**
 * Single-pass aggregator `Flow<PromptExecutionEvent>` → `PromptExecutionResult`. Used where
 * a final snapshot of the model's response is needed (serializers, agents, tests,
 * standalone pipeline stages). Direct consumers of the flow usually collect locally just
 * the events they care about and don't call this.
 *
 * Flows are cold, double-collection would re-execute the request — the aggregator is a
 * single source of truth for both the response payload **and** the final
 * [PromptExecutionStatistics]. Provider only emits primitive events (started / part-received
 * / usage-delta / completed / failed); the aggregator combines them into a fully populated
 * statistics object including `responseMessage` (from the accumulated [MessagePart] list).
 *
 * Event → result-variant mapping:
 * - any `MessagePart.ToolCall` seen → [ToolCallPromptExecutionResult];
 * - any `MessagePart.File` seen → [ImagePromptExecutionResult.File] (first URI wins);
 * - any `MessagePart.InlineData` seen → [ImagePromptExecutionResult.Bytes] (first blob);
 * - otherwise → [TextPromptExecutionResult].
 *
 * If both `File` and `InlineData` parts appeared, `File` wins — the storage decorator
 * transformed the inline blob, the bytes are redundant and likely empty.
 *
 * If a `Failed` event arrives the error is rethrown after the aggregator finalises.
 */
suspend fun Flow<PromptExecutionEvent>.aggregate(): PromptExecutionResult {
    val collectedParts = mutableListOf<MessagePart>()
    val textParts = mutableListOf<ExecutionTextPart>()
    val toolCalls = mutableListOf<ToolCall>()
    val inlineParts = mutableListOf<MessagePart.InlineData>()
    val fileParts = mutableListOf<MessagePart.File>()
    val embeddingParts = mutableListOf<MessagePart.Embedding>()

    var started: PromptExecutionEvent.PromptStarted? = null
    var servedModelName: String? = null
    var inputTotal = InputTokenStatistics.EMPTY
    var outputTotal = OutputTokenStatistics.EMPTY
    var finishReason: String? = null
    var success = true
    var errorMessage: String? = null
    var failure: Throwable? = null
    var streamDiagnostics: StreamDiagnostics? = null

    fun statistics(): PromptExecutionStatistics {
        val responseMessage = if (collectedParts.isEmpty()) null
        else PromptMessage(role = PromptContentType.MODEL, parts = collectedParts.toList())
        return buildStatistics(started, servedModelName, inputTotal, outputTotal, responseMessage, success, errorMessage, finishReason, streamDiagnostics)
    }

    try {
        collect { event ->
            when (event) {
                is PromptExecutionEvent.PromptStarted -> started = event
                is PromptExecutionEvent.StreamResponded -> {
                    check(started != null) { "StreamResponded arrived before PromptStarted" }
                    servedModelName = event.servedModelName
                }
                is PromptExecutionEvent.PartReceived -> {
                    val part = event.part
                    collectedParts += part
                    when (part) {
                        is MessagePart.Text -> textParts += ExecutionTextPart(part.text, isThought = false)
                        is MessagePart.Thought -> textParts += ExecutionTextPart(part.text, part.signature, isThought = true)
                        is MessagePart.ToolCall -> toolCalls += part.toToolCall()
                        is MessagePart.InlineData -> inlineParts += part
                        is MessagePart.File -> fileParts += part
                        is MessagePart.Embedding -> embeddingParts += part
                        is MessagePart.ToolResult -> Unit
                    }
                }
                is PromptExecutionEvent.UsageDelta -> {
                    inputTotal = inputTotal + event.input
                    outputTotal = outputTotal + event.output
                }
                is PromptExecutionEvent.Completed -> {
                    finishReason = event.finishReason
                    streamDiagnostics = event.streamDiagnostics
                }
                is PromptExecutionEvent.Failed -> {
                    success = false
                    errorMessage = event.error.message
                    finishReason = event.finishReason
                    streamDiagnostics = event.streamDiagnostics
                    failure = event.error
                }
                // CacheCreated — side-channel event consumed by the agent via `Flow.onEach`
                // (see `TextPromptAgent`); for round-trip aggregation it's a no-op.
                is PromptExecutionEvent.CacheCreated -> {}
            }
        }
    } catch (e: PromptExecutionException) {
        // Providers throw after (or without) a `Failed` event, so `collect` never returns normally:
        // the statistics are attached here, through the same `buildStatistics` as in `ToolCallingAgent`.
        // Only when `PromptStarted` arrived; before that there is genuinely nothing to carry.
        if (success) {
            success = false
            errorMessage = e.message
        }
        if (started != null) e.partialStats = statistics()
        throw e
    }

    val finalStats = statistics()


    failure?.let { f ->
        // Carry the fully-assembled statistics out through the exception — tokens,
        // responseMessage, model name, everything `finalStats` already holds — so the failure
        // statistics rebuilt in agent catch-blocks reflect the round-trip's real partial content
        // instead of an empty stub. Only when something actually arrived (at least
        // `PromptStarted`, which precedes the request) — keep `partialStats` null when there is genuinely nothing to carry.
        if (started != null) {
            (f as? PromptExecutionException)?.partialStats = finalStats
        }
        throw f
    }

    val firstFile = fileParts.firstOrNull()
    val firstInline = inlineParts.firstOrNull()

    return when {
        embeddingParts.isNotEmpty() -> EmbeddingPromptExecutionResult(embeddingParts.first().values, finalStats)
        toolCalls.isNotEmpty() -> ToolCallPromptExecutionResult(toolCalls, finalStats, textParts)
        firstFile != null -> ImagePromptExecutionResult.File(
            uri = firstFile.uri,
            mimeType = firstFile.mimeType ?: "application/octet-stream",
            statistics = finalStats,
        )
        firstInline != null -> ImagePromptExecutionResult.Bytes(
            imageBytes = firstInline.data ?: byteArrayOf(),
            mimeType = firstInline.mimeType,
            statistics = finalStats,
        )
        else -> TextPromptExecutionResult(finalStats, textParts)
    }
}

/**
 * The single mapping from accumulated round-trip fields to [PromptExecutionStatistics]; used by
 * [aggregate] and by `ToolCallingAgent` (success and failure paths). The folding loops stay
 * separate (the agent's is interleaved with side effects), the field mapping does not.
 */
internal fun buildStatistics(
    started: PromptExecutionEvent.PromptStarted?,
    servedModelName: String?,
    input: InputTokenStatistics,
    outputTotal: OutputTokenStatistics,
    responseMessage: PromptMessage?,
    success: Boolean,
    errorMessage: String?,
    finishReason: String?,
    streamDiagnostics: StreamDiagnostics?,
): PromptExecutionStatistics {
    val cacheStats = started?.cacheStats
    return PromptExecutionStatistics(
        modelName = servedModelName ?: started?.modelName ?: "",
        requestedModelName = started?.requestedModelName,
        attempts = started?.attempts ?: 1,
        success = success,
        errorMessage = errorMessage,
        tokens = PromptExecutionTokenStatistics(
            totalTokens = input.total + outputTotal.total,
            input = input,
            output = outputTotal,
        ),
        requestMessages = started?.requestMessages.orEmpty(),
        responseMessage = responseMessage,
        cacheId = started?.cacheId,
        cachedTokenCount = cacheStats?.tokenCount,
        isCacheCreated = cacheStats?.isCreated,
        cacheTtlMillis = cacheStats?.ttl?.inWholeMilliseconds,
        isCacheExpired = cacheStats?.isExpired,
        cacheExpiredAtMillis = cacheStats?.expiredAt?.toEpochMilliseconds(),
        cacheScope = cacheStats?.scope,
        finishReason = finishReason,
        streamDiagnostics = streamDiagnostics,
    )
}

/**
 * Reverse helper of [aggregate]: expand a [PromptExecutionResult] back into a flow of
 * events. Used in test mocks to express the expected model response in the familiar
 * `*PromptExecutionResult` shape and have the matching event sequence emitted for free.
 */
fun PromptExecutionResult.toEventFlow(): Flow<PromptExecutionEvent> = flow {
    val stats = when (val r = this@toEventFlow) {
        is TextPromptExecutionResult -> r.statistics
        is ToolCallPromptExecutionResult -> r.statistics
        is ImagePromptExecutionResult -> r.statistics
        is EmbeddingPromptExecutionResult -> r.statistics
    }
    emit(PromptExecutionEvent.PromptStarted(
        modelName = stats.modelName,
        requestedModelName = stats.requestedModelName,
        requestMessages = stats.requestMessages,
        cacheId = stats.cacheId,
        attempts = stats.attempts,
    ))
    emit(PromptExecutionEvent.StreamResponded(servedModelName = stats.modelName))
    when (val result = this@toEventFlow) {
        is TextPromptExecutionResult -> {
            for (part in result.textParts) emit(PromptExecutionEvent.PartReceived(part.toMessagePart()))
        }
        is ToolCallPromptExecutionResult -> {
            for (part in result.textParts) emit(PromptExecutionEvent.PartReceived(part.toMessagePart()))
            for (call in result.toolCalls) emit(PromptExecutionEvent.PartReceived(call.toMessagePart()))
        }
        is ImagePromptExecutionResult.Bytes -> emit(PromptExecutionEvent.PartReceived(
            MessagePart.InlineData(
                mimeType = result.mimeType,
                sizeBytes = result.imageBytes.size.toLong(),
                data = result.imageBytes,
            )
        ))
        is ImagePromptExecutionResult.File -> emit(PromptExecutionEvent.PartReceived(
            MessagePart.File(uri = result.uri, mimeType = result.mimeType)
        ))
        is EmbeddingPromptExecutionResult -> emit(PromptExecutionEvent.PartReceived(
            MessagePart.Embedding(result.embedding.size, result.embedding)
        ))
    }
    if (stats.tokens.input != InputTokenStatistics.EMPTY || stats.tokens.output != OutputTokenStatistics.EMPTY)
        emit(PromptExecutionEvent.UsageDelta(input = stats.tokens.input, output = stats.tokens.output))
    if (stats.success) emit(PromptExecutionEvent.Completed(stats.finishReason, stats.streamDiagnostics))
    else emit(PromptExecutionEvent.Failed(
        RuntimeException(stats.errorMessage ?: "unspecified failure"),
        stats.finishReason,
        stats.streamDiagnostics,
    ))
}

private fun ExecutionTextPart.toMessagePart(): MessagePart =
    if (isThought) MessagePart.Thought(text, thoughtSignature) else MessagePart.Text(text)
