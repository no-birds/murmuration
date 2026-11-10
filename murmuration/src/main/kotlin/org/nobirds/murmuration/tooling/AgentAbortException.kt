package org.nobirds.murmuration.tooling

/**
 * Thrown to abort the whole agent run instead of reporting a failure to the model.
 *
 * Any other exception from a tool is turned into an `"Error: …"` result the model reads and reacts
 * to; an [AgentAbortException] is not: it leaves the agentic loop as is, the model gets no further
 * round-trip. Use it for a failure the model must not paper over — one that cancels the surrounding
 * work as a whole.
 *
 * Contract:
 *  1. An abort is legal from ANY code the agentic loop runs on behalf of the run: a tool body, the
 *     [Stateful] hooks (`onAfterToolCalls`, `onRoundTripAborted`) and round-trip buffers inside
 *     them. It may arrive wrapped in a foreign exception.
 *  2. One predicate decides "is this an abort?" everywhere: [org.nobirds.murmuration.retry.abortsAgent]
 *     (the cause chain) — at both retry decision points and in every `catch` of the loop that would
 *     otherwise turn the exception into a model-visible result, a WARN or a continuation.
 *  3. Such a `catch` rethrows the original exception unchanged
 *     ([org.nobirds.murmuration.retry.rethrowIfAborts]): the wrapper and the [cause] survive, the
 *     operator needs them to see why the run was aborted. An abort from a tool-job outranks a
 *     failed stream's error.
 *
 * The abort is terminal for every framework retry: [org.nobirds.murmuration.retry.abortsAgent]
 * is checked by `RetrySession.nextDelayOrNull` and `retryWithTimeout` BEFORE the retry strategy is
 * consulted, so a retryable [cause] (transient, cache-expired, I/O) does not get the abort replayed.
 * The [cause] is preserved so the operator sees why the run was aborted.
 */
open class AgentAbortException(
    message: String,
    cause: Throwable? = null,
) : RuntimeException(message, cause)
