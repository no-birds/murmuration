package org.nobirds.murmuration.gemini

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject

/**
 * Canonical key position -> wire name for one tool. A position is the path of canonical keys from the
 * root of the arguments; array elements and `anyOf` variants share the path of their container, so the
 * map never needs to know which variant a value belongs to. Keys without an entry stay canonical.
 */
class WireKeyMap(private val wireNameByPath: Map<List<String>, String>) {

    val isEmpty: Boolean get() = wireNameByPath.isEmpty()

    /** Wire name for the key at [path] (parent path + canonical key), or `null` when it stays canonical. */
    fun wireNameAt(path: List<String>): String? = wireNameByPath[path]

    /** Canonical arguments to wire arguments, strictly by this map; the schema is never consulted. */
    fun toWire(args: Map<String, JsonElement>): Map<String, JsonElement> =
        if (isEmpty) args else (JsonObject(args).toWire(emptyList()) as JsonObject).toMap()

    private fun JsonElement.toWire(path: List<String>): JsonElement = when (this) {
        is JsonObject -> JsonObject(entries.associate { (key, value) ->
            val keyPath = path + key
            (wireNameByPath[keyPath] ?: key) to value.toWire(keyPath)
        })
        is JsonArray -> JsonArray(map { it.toWire(path) })
        else -> this
    }

    companion object {
        val EMPTY = WireKeyMap(emptyMap())
    }
}
