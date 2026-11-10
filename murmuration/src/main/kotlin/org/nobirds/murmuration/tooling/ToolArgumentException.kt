package org.nobirds.murmuration.tooling

/**
 * The model's tool-call arguments do not fit the tool's declared parameter types: an unknown enum
 * value, a wrong shape, an unparsable primitive. Thrown by [ToolArgumentDecoder] before the tool is
 * invoked, carrying EVERY problem of the call in [problems] (one refusal, not one per round-trip).
 *
 * It is an expected, model-side mistake, not a framework failure: the agent loop turns it into the
 * usual `"Error: …"` tool result and logs it as a WARN without a stack trace.
 */
class ToolArgumentException(val problems: List<String>) : IllegalArgumentException(problems.joinToString("; ")) {
    init {
        require(problems.isNotEmpty()) { "ToolArgumentException needs at least one problem" }
    }
}
