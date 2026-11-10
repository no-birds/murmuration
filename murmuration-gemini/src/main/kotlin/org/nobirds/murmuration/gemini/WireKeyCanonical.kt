package org.nobirds.murmuration.gemini

import org.nobirds.murmuration.prompt.CallIds
import com.google.genai.types.FunctionCall
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.toJsonElement
import kotlin.jvm.optionals.getOrNull

/** Marker that starts every wire (tail-aligned) argument key. A canonical key never starts with it. */
internal const val WIRE_KEY_MARKER: Char = '~'

private const val WIRE_KEY_PREFIX_WIDTH: Int = 1 + GeminiWireKeyTable.SUPPORTED_WIDTH

/**
 * Lexical inverse of the wire layout: a key that starts with [WIRE_KEY_MARKER] and is longer than the
 * prefix loses the `~`+code prefix; any other key is unchanged. Works on every depth (objects, arrays).
 * Looks at neither schema nor table, so re-applying it to canonical names is a no-op.
 */
fun JsonElement.toCanonical(): JsonElement = when (this) {
    is JsonObject -> JsonObject(entries.associate { (k, v) -> k.toCanonicalKey() to v.toCanonical() })
    is JsonArray -> JsonArray(map { it.toCanonical() })
    else -> this
}

fun String.toCanonicalKey(): String =
    if (length > WIRE_KEY_PREFIX_WIDTH && this[0] == WIRE_KEY_MARKER) substring(WIRE_KEY_PREFIX_WIDTH) else this

internal fun Map<String, JsonElement>.toCanonicalArgs(): Map<String, JsonElement> =
    JsonObject(this).toCanonical().let { (it as JsonObject).toMap() }

/**
 * The single place that builds a [MessagePart.ToolCall] from a genai [FunctionCall]: used by the request
 * telemetry mapping and by the response event mapping. Argument keys are brought to canonical names here.
 */
internal fun FunctionCall.toToolCall(thoughtSignature: String?): MessagePart.ToolCall =
    MessagePart.ToolCall(
        name = name().getOrNull() ?: "",
        args = args().getOrNull()
            ?.mapValues { (_, v) -> v.toJsonElement() }
            ?.toCanonicalArgs()
            ?: emptyMap(),
        thoughtSignature = thoughtSignature,
        callId = CallIds.mint(),
    )
