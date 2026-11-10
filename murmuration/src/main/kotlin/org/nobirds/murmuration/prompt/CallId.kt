package org.nobirds.murmuration.prompt

import java.util.UUID
import kotlinx.serialization.Serializable

/**
 * Identity of one tool call, carried unchanged from the model's call to the result the agent sends
 * back (`ToolCall` -> `ToolOutcome` -> `ToolResult`). Never null in the core.
 *
 * Provider mappers use it to pair a call with its result on the wire; a provider that has its own
 * id (OpenAI `call_id`, Anthropic `tool_use.id`) reads it from the response, otherwise one is
 * minted with [CallIds.mint].
 */
@JvmInline
@Serializable
value class CallId(val value: String) {
    override fun toString(): String = value

    /** True for [LEGACY], i.e. a record that predates call ids. */
    val isLegacy: Boolean get() = value.isEmpty()

    companion object {
        /**
         * Placeholder for records written before call ids existed. It exists ONLY so those records
         * can be deserialized and displayed: it must never be minted for new calls and a provider
         * mapper must refuse to send it (an `InvalidRequestException` before the network).
         */
        val LEGACY: CallId = CallId("")
    }
}

/** Source of fresh, globally unique [CallId]s. */
object CallIds {
    fun mint(): CallId = CallId(UUID.randomUUID().toString())
}
