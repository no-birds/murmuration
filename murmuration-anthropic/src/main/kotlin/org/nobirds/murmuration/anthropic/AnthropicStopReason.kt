package org.nobirds.murmuration.anthropic

import org.nobirds.murmuration.prompt.MaxTokensException
import org.nobirds.murmuration.prompt.PolicyCategory
import org.nobirds.murmuration.prompt.PolicyViolationException
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionException
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.UnknownProviderException

/**
 * Judges a finished Messages API response: the raw `stop_reason` of the stream plus whether it produced any content.
 * Returns `null` for an acceptable response, otherwise the typed core exception the executor must raise.
 *
 * - `end_turn`, `stop_sequence`, `tool_use` — normal ends;
 * - `refusal` — [PolicyViolationException] ([PolicyCategory.OTHER], the raw token kept): retrying the same prompt gives
 *   the same refusal, the fix is a prompt rewrite;
 * - `max_tokens` — [MaxTokensException] (retryable in the core: another thinking trajectory may fit the cap);
 * - no `stop_reason` at all — `TransientException(INCOMPLETE_RESPONSE)`;
 * - any other value (`pause_turn` belongs to server tools this provider never enables) — [UnknownProviderException],
 *   loud and not retried: silently accepting an unfinished turn is worse.
 *
 * A normally finished response without content (no text, no thought, no tool call) is
 * `TransientException(EMPTY_RESPONSE)`: it expresses no intention of the model, so it is a provider error, not a turn.
 * Truncation and refusal are judged first — an empty answer under `max_tokens` is a truncation (the cap was eaten by
 * thinking), not an "empty" one.
 */
internal fun judgeStopReason(prompt: Prompt, stopReason: String?, producedContent: Boolean): PromptExecutionException? =
    when (stopReason) {
        null -> TransientException(TransientReason.INCOMPLETE_RESPONSE, "Anthropic response ended without a stop_reason", null)
        "refusal" -> PolicyViolationException(prompt, PolicyCategory.OTHER, stopReason, "Claude refused to answer (stop_reason=refusal)", null)
        "max_tokens" -> MaxTokensException(prompt, "Response truncated: stop_reason=max_tokens", null)
        "end_turn", "stop_sequence", "tool_use" ->
            if (producedContent) null
            else TransientException(TransientReason.EMPTY_RESPONSE, "Anthropic returned an empty response (stop_reason=$stopReason, no content)", null)
        else -> UnknownProviderException("StopReason.$stopReason", "Unexpected stop_reason: $stopReason", null)
    }
