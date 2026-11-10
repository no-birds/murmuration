package org.nobirds.murmuration.anthropic

/**
 * The opaque thought carrier of the core (`TextPromptPart.thoughtSignature` of a THOUGHT part) as the
 * Anthropic provider reads and writes it. The core stores one string and never looks inside; the kind of
 * wire block is encoded here, inside the provider.
 *
 * - a `thinking` block: the carrier is its `signature`, verbatim, and the part text is the thinking text;
 * - a `redacted_thinking` block: the carrier is [REDACTED_PREFIX] + its `data`, verbatim, the part text is empty.
 *   The prefix can never start a real signature (a signature is base64, which has no colon), so the two kinds
 *   never collide.
 */
internal sealed interface AnthropicThoughtCarrier {

    /** Signature of a `thinking` block. */
    data class Signature(val signature: String) : AnthropicThoughtCarrier

    /** Opaque payload of a `redacted_thinking` block. */
    data class Redacted(val data: String) : AnthropicThoughtCarrier

    companion object {
        const val REDACTED_PREFIX: String = "redacted:"

        /** Carrier string of a `thinking` block. */
        fun encodeSignature(signature: String): String = signature

        /** Carrier string of a `redacted_thinking` block. */
        fun encodeRedacted(data: String): String = REDACTED_PREFIX + data

        /** Reads a carrier string; `null` when there is no carrier (null or empty) — such a thought is not replayable. */
        fun decode(carrier: String?): AnthropicThoughtCarrier? = when {
            carrier.isNullOrEmpty() -> null
            carrier.startsWith(REDACTED_PREFIX) -> Redacted(carrier.removePrefix(REDACTED_PREFIX))
            else -> Signature(carrier)
        }
    }
}
