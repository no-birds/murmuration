package org.nobirds.murmuration.tooling

import kotlinx.coroutines.ThreadContextElement
import kotlinx.coroutines.currentCoroutineContext
import kotlin.coroutines.AbstractCoroutineContextElement
import kotlin.coroutines.CoroutineContext

/**
 * Publishes the name of the tool call that is executing right now into the coroutine context,
 * so a tool body (and anything it calls) can ask "which tool am I running as" without a new
 * parameter channel. The name is the tool's name in the schema — the same one [instrumentName]
 * computes — because the executor resolves the call by that name before invoking it.
 *
 * Installed by the tool executor around exactly one invocation: two calls of one batch run in
 * their own coroutines (parallel or sequential), each sees its own element, and nothing leaks to
 * code outside the call. Outside a tool call the element is absent ([currentToolName] is null).
 *
 * The element is also a [ThreadContextElement]: while the call's coroutine runs on a thread, the same name is visible
 * to plain (non-suspend) code on that thread through [currentToolNameOnThread]. Tool bodies are ordinary functions, so
 * this is the reader they can call; it is restored on every suspension and resumption, so it never leaks to another call.
 */
class CurrentToolCall(val toolName: String) : AbstractCoroutineContextElement(CurrentToolCall), ThreadContextElement<String?> {
    companion object Key : CoroutineContext.Key<CurrentToolCall>

    override fun updateThreadContext(context: CoroutineContext): String? =
        nameOnThread.get().also { nameOnThread.set(toolName) }

    override fun restoreThreadContext(context: CoroutineContext, oldState: String?) = nameOnThread.set(oldState)
}

private val nameOnThread = ThreadLocal<String?>()

/** Name of the tool whose body is executing in the current coroutine, or null outside a tool call. */
suspend fun currentToolName(): String? = currentCoroutineContext()[CurrentToolCall]?.toolName

/**
 * Name of the tool whose call is executing on the current thread, or null outside a tool call. The reader for plain
 * (non-suspend) tool bodies; the same fact as [currentToolName], kept by the same element.
 */
fun currentToolNameOnThread(): String? = nameOnThread.get()
