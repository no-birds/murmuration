package org.nobirds.murmuration.prompt

import kotlinx.coroutines.flow.Flow
import org.nobirds.murmuration.llm.Llm

/**
 * Selects the concrete [PromptExecutor] for a prompt by its [Llm] — each delegate declares what it
 * serves via [PromptExecutor.supports] (e.g. a Gemini executor: `provider in {vertex, google}`).
 *
 * Sits at the base, under the provider-agnostic decorators (binary-storage, throttling, prompt
 * settings) — those stay shared above it. With one provider this is a degenerate guard; adding a
 * provider is one more delegate, and callers keep calling [execute] unchanged. When no delegate
 * accepts the model it fails loudly rather than silently feeding it to the wrong provider.
 */
class RoutingPromptExecutor(private val executors: List<PromptExecutor>) : PromptExecutor {
    override fun execute(prompt: Prompt): Flow<PromptExecutionEvent> =
        (executors.firstOrNull { it.supports(prompt.llm) }
            ?: error("No PromptExecutor supports provider '${prompt.llm.provider}' (model '${prompt.llm.model}')"))
            .execute(prompt)

    override fun supports(llm: Llm): Boolean = executors.any { it.supports(llm) }
}
