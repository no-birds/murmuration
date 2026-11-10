package org.nobirds.murmuration.tooling

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import org.nobirds.murmuration.prompt.ToolCall

/**
 * Executes a single tool call. Real implementation — `ToolsExecutor.executeCall` in
 * `ToolCallingAgent.kt` (looks up `@Instrument`, deserialises args, invokes setter,
 * sanitises result). Session wrappers implement the same interface with additional
 * scheduling logic (serialisation / parallelisation).
 *
 * Carries [ToolOutcome] — the fact of how the call was resolved — not the prompt part built
 * from it. The projection into [org.nobirds.murmuration.prompt.ToolResultPromptPart] happens in
 * exactly one place, on batch assembly (`ToolCallingAgent.executeTurn`), not on this channel.
 */
fun interface ToolExecutor {
    suspend fun execute(call: ToolCall): ToolOutcome
}

/**
 * Scheduling strategy for tool calls within a single LLM round-trip. Caller wraps an
 * underlying [ToolExecutor] via [open] and submits calls via the returned
 * [ScopedToolExecutor]; the scheduler decides whether they run serially or in parallel.
 * After streaming finishes, caller must [ScopedToolExecutor.close] to release resources.
 *
 * Default — [SequentialToolCallScheduler] — serialises calls through a [Channel] +
 * single collector. Required because pipeline builders mutate shared state through
 * non-atomic `shared.X = shared.X.copy(...)` (bug 2026-05-19: parallel `addClass` +
 * `updateBasicInfo` raced on `shared.base`, lost the `classes` field).
 *
 * [ParallelToolCallScheduler] — legacy fan-out: `execute` goes straight to the underlying
 * executor without serialisation. Use only when callers can guarantee that closures touch
 * disjoint state.
 */
interface ToolCallScheduler {
    fun open(scope: CoroutineScope, executor: ToolExecutor): ScopedToolExecutor
}

/**
 * [ToolExecutor] with a tied lifecycle — caller must call [close] once submissions
 * stop. `execute` itself usually suspends the caller until the scheduler has processed it; to
 * overlap with the LLM stream the caller wraps `execute` in `async{}`.
 */
interface ScopedToolExecutor : ToolExecutor {
    suspend fun close()
}

/** Suspend analogue of `AutoCloseable.use` — guarantees close() even if the block throws. */
suspend inline fun <T> ScopedToolExecutor.use(block: (ScopedToolExecutor) -> T): T {
    try {
        return block(this)
    } finally {
        close()
    }
}

class SequentialToolCallScheduler : ToolCallScheduler {
    override fun open(scope: CoroutineScope, executor: ToolExecutor): ScopedToolExecutor =
        SequentialSession(scope, executor)

    private class SequentialSession(
        scope: CoroutineScope,
        private val executor: ToolExecutor,
    ) : ScopedToolExecutor {
        // Channel + single collector: processes items strictly one at a time. Submit
        // puts a CompletableDeferred into the queue and awaits. Close closes the channel +
        // waits for the collector (it exits the for loop after draining the rest).
        private val channel = Channel<Entry>(Channel.UNLIMITED)
        private val collector: Job = scope.launch {
            for ((call, deferred) in channel) {
                try {
                    deferred.complete(executor.execute(call))
                } catch (e: Throwable) {
                    deferred.completeExceptionally(e)
                }
            }
        }

        override suspend fun execute(call: ToolCall): ToolOutcome {
            val deferred = CompletableDeferred<ToolOutcome>()
            channel.send(Entry(call, deferred))
            return deferred.await()
        }

        override suspend fun close() {
            channel.close()
            collector.join()
        }

        private data class Entry(
            val call: ToolCall,
            val deferred: CompletableDeferred<ToolOutcome>,
        )
    }
}

class ParallelToolCallScheduler : ToolCallScheduler {
    override fun open(scope: CoroutineScope, executor: ToolExecutor): ScopedToolExecutor =
        ParallelSession(executor)

    private class ParallelSession(private val executor: ToolExecutor) : ScopedToolExecutor {
        // Just proxies. Parallelism is provided by the caller via async{} around execute().
        override suspend fun execute(call: ToolCall): ToolOutcome = executor.execute(call)
        override suspend fun close() {} // no resources
    }
}
