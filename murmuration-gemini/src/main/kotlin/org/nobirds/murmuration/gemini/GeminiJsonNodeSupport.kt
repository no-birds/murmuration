package org.nobirds.murmuration.gemini

import com.fasterxml.jackson.databind.JsonNode
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

fun JsonNode.toJsonElement(): JsonElement = when {
    isTextual -> JsonPrimitive(textValue())
    isBoolean -> JsonPrimitive(booleanValue())
    isNumber -> JsonPrimitive(numberValue())
    isNull -> JsonNull
    isArray -> JsonArray(map { it.toJsonElement() })
    isObject -> JsonObject(properties().associate { it.key to it.value.toJsonElement() })
    else -> JsonPrimitive(toString())
}
