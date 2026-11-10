package org.nobirds.murmuration.agent

class PipelineEventListenerBuilder internal constructor() {
    private val listeners = mutableListOf<PipelineEventListener>()

    /** Add a listener. `null` is ignored — convenient for optional listeners (e.g. progress only when gameId is known). */
    fun add(listener: PipelineEventListener?): PipelineEventListenerBuilder = apply {
        listener?.let { listeners.add(it) }
    }

    internal fun build(): PipelineEventListener = when (listeners.size) {
        0 -> EmptyPipelineEventListener
        1 -> listeners.single()
        else -> CompositePipelineEventListener(listeners.toList())
    }
}

/**
 * Build a [PipelineEventListener] that fans out to several listeners. Nullable adds are
 * skipped, so the same call site can wire optional progress listeners without `if`:
 *
 * ```
 * val listener = pipelineEventListener {
 *     add(collector.withNotifications(...))
 *     add(progressListener)  // may be null
 * }
 * ```
 *
 * If exactly one listener is added the builder returns it as-is (zero overhead). If none —
 * returns [EmptyPipelineEventListener].
 */
fun pipelineEventListener(block: PipelineEventListenerBuilder.() -> Unit): PipelineEventListener =
    PipelineEventListenerBuilder().apply(block).build()

object EmptyPipelineEventListener : PipelineEventListener
