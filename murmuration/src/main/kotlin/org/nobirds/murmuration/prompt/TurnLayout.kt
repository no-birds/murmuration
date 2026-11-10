package org.nobirds.murmuration.prompt

/**
 * Wire layout of a model turn: non-call parts first, then tool calls, each group in its original
 * order (a stable partition). The agent keeps a turn in arrival order; providers whose protocol
 * wants calls last apply this in their mappers. Idempotent; a turn without calls is returned as is.
 */
fun List<PromptPart>.callsLast(): List<PromptPart> {
    val (calls, rest) = partition { it is ToolCallPromptPart }
    return if (calls.isEmpty()) this else rest + calls
}

/** Applies [callsLast] to the MODEL contents; other contents are untouched. */
fun List<PromptContent>.withCallsLast(): List<PromptContent> = map { content ->
    if (content.type != PromptContentType.MODEL) content
    else {
        val laidOut = content.parts.callsLast()
        if (laidOut === content.parts) content else PromptContent(content.type, laidOut)
    }
}
