package org.nobirds.murmuration.prompt

/**
 * The single place where a streamed [MessagePart.ToolCall] is converted to an executable [ToolCall]
 * and back. Every field is carried over as is; keep both directions in sync when [ToolCall] grows.
 */
fun MessagePart.ToolCall.toToolCall(): ToolCall = ToolCall(name, args, thoughtSignature, callId)

/** Inverse of [toToolCall]. */
fun ToolCall.toMessagePart(): MessagePart.ToolCall = MessagePart.ToolCall(name, args, thoughtSignature, callId)
