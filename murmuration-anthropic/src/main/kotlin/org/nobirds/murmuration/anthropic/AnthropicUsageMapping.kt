package org.nobirds.murmuration.anthropic

import com.anthropic.models.messages.MessageDeltaUsage
import com.anthropic.models.messages.Usage
import org.nobirds.murmuration.prompt.InputTokenStatistics
import org.nobirds.murmuration.prompt.OutputTokenStatistics
import org.nobirds.murmuration.prompt.PromptExecutionEvent.UsageDelta
import org.nobirds.murmuration.prompt.UnknownProviderException
import org.nobirds.murmuration.prompt.minus
import kotlin.jvm.optionals.getOrNull

/** Marker `rawClass` of the typed usage-mapping error: it is not retried (the retry strategy only retries what it lists). */
internal const val USAGE_MAPPING_ERROR: String = "anthropic.usage-mapping"

private fun usageMappingError(message: String): UnknownProviderException =
    UnknownProviderException(USAGE_MAPPING_ERROR, message, null)

/**
 * Input statistics of a `message_start` [Usage]. Claude reports the three input classes separately and
 * `input_tokens` excludes both cache classes, so `total = input + cache_read + cache_creation`.
 *
 * The class of a cache write is never guessed: `cache_creation_input_tokens` must be explained exactly by the
 * `cache_creation` breakdown (`ephemeral_5m + ephemeral_1h`). A write without a breakdown, or a breakdown that does
 * not add up to the aggregate, is a mapping error — a difference never falls silently into regular input.
 */
internal fun Usage.toInputStatistics(): InputTokenStatistics {
    val read = cacheReadInputTokens().getOrNull() ?: 0L
    val creation = cacheCreationInputTokens().getOrNull() ?: 0L
    val breakdown = cacheCreation().getOrNull()
    if (breakdown == null && creation > 0) {
        throw usageMappingError(
            "Anthropic usage reports cache_creation_input_tokens=$creation without the cache_creation TTL breakdown; the class of the write cannot be guessed"
        )
    }
    val short = breakdown?.ephemeral5mInputTokens() ?: 0L
    val long = breakdown?.ephemeral1hInputTokens() ?: 0L
    if (short + long != creation) {
        throw usageMappingError(
            "Anthropic usage cache_creation breakdown (5m=$short, 1h=$long) does not add up to cache_creation_input_tokens=$creation"
        )
    }
    return InputTokenStatistics(
        total = inputTokens() + read + creation,
        cached = read,
        cacheWriteShort = short,
        cacheWriteLong = long,
    )
}

/**
 * Output statistics: `output_tokens` includes the thinking tokens, the core sums `total` and `thinking` as two
 * lines, so `total = output_tokens − thinking_tokens` and `thinking = thinking_tokens` (no double count; same as OpenAI).
 */
internal fun outputStatistics(outputTokens: Long, thinkingTokens: Long): OutputTokenStatistics =
    OutputTokenStatistics(total = outputTokens - thinkingTokens, thinking = thinkingTokens)

/**
 * Turns the usage of one stream into core [UsageDelta]s (the core sums increments).
 *
 * - `message_start` carries the input side (the only place with the `cache_creation` breakdown) and the first output count;
 * - `message_delta` usage is cumulative for output: only the increment over the last emitted value is sent.
 *   Its input-side fields are not read — the input classes are fixed at `message_start`, and the TTL breakdown that
 *   makes them mappable is absent from `message_delta`.
 */
internal class AnthropicUsageTracker {
    private var emittedOutput: OutputTokenStatistics = OutputTokenStatistics.EMPTY

    fun onMessageStart(usage: Usage): UsageDelta {
        val output = outputStatistics(usage.outputTokens(), usage.outputTokensDetails().getOrNull()?.thinkingTokens() ?: 0L)
        emittedOutput = output
        return UsageDelta(input = usage.toInputStatistics(), output = output)
    }

    fun onMessageDelta(usage: MessageDeltaUsage): UsageDelta {
        val cumulative = outputStatistics(usage.outputTokens(), usage.outputTokensDetails().getOrNull()?.thinkingTokens() ?: 0L)
        val increment = cumulative - emittedOutput
        emittedOutput = cumulative
        return UsageDelta(output = increment)
    }
}
