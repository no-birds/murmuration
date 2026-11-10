package org.nobirds.murmuration.prompt

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.tooling.ToolOutcome

/**
 * The stored trace (`MessagePart`) does not carry [CallId] and reads it back as [CallId.LEGACY];
 * the file dumps (`ToolCall`, `ToolOutcome`) carry it and read a missing field as [CallId.LEGACY].
 */
class CallIdSerializationTest {
    private val json = Json { encodeDefaults = true; ignoreUnknownKeys = true }
    private val id = CallId("c-1")

    @Test
    fun `message tool_call is written without callId and read as LEGACY`() {
        val part: MessagePart = MessagePart.ToolCall("ping", mapOf("a" to JsonPrimitive(1)), "sig", id)
        val encoded = json.encodeToString(MessagePart.serializer(), part)
        assertEquals("""{"type":"tool_call","name":"ping","args":{"a":1},"thoughtSignature":"sig"}""", encoded)
        val back = json.decodeFromString(MessagePart.serializer(), encoded) as MessagePart.ToolCall
        assertTrue(back.callId.isLegacy)
        assertEquals(part, back.copy(callId = id))
    }

    @Test
    fun `message tool_result is written without callId and read as LEGACY`() {
        val part: MessagePart = MessagePart.ToolResult("ping", JsonPrimitive("pong"), null, id)
        val encoded = json.encodeToString(MessagePart.serializer(), part)
        assertFalse("callId" in encoded)
        val back = json.decodeFromString(MessagePart.serializer(), encoded) as MessagePart.ToolResult
        assertEquals(CallId.LEGACY, back.callId)
    }

    @Test
    fun `ToolCall dump round-trips callId and reads a missing field as LEGACY`() {
        val call = ToolCall("ping", emptyMap(), null, id)
        assertEquals(call, json.decodeFromString(ToolCall.serializer(), json.encodeToString(ToolCall.serializer(), call)))
        val old = json.decodeFromString(ToolCall.serializer(), """{"name":"ping","args":{}}""")
        assertEquals(CallId.LEGACY, old.callId)
    }

    @Test
    fun `ToolOutcome dump round-trips callId and reads a missing field as LEGACY`() {
        val outcome = ToolOutcome(
            "ping", emptyMap(), JsonPrimitive("pong"), false, 0, true, ToolOutcome.Exit.INVOKED, null, null, id,
        )
        val encoded = json.encodeToString(ToolOutcome.serializer(), outcome)
        assertEquals(outcome, json.decodeFromString(ToolOutcome.serializer(), encoded))
        val old = """{"name":"ping","args":{},"rendered":"pong","isError":false,"ordinal":0,"invoked":true,"exit":"INVOKED","declaredReader":null}"""
        assertEquals(CallId.LEGACY, json.decodeFromString(ToolOutcome.serializer(), old).callId)
    }

    @Test
    fun `conversion carries callId in both directions`() {
        val part = MessagePart.ToolCall("ping", emptyMap(), null, id)
        assertEquals(id, part.toToolCall().callId)
        assertEquals(part, part.toToolCall().toMessagePart())
    }
}
