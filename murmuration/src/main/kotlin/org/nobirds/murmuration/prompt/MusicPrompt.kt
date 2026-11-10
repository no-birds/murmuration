package org.nobirds.murmuration.prompt

import org.nobirds.murmuration.agent.PromptExecutorException
import org.nobirds.murmuration.llm.Llm

/**
 * Convenience: generate music from a scene [description] through this executor directly — no agent
 * layer, so no [org.nobirds.murmuration.agent.ExecutionContext] is required. Music is generated
 * lazily at stream time (outside a pipeline), so the direct path fits; when generation should land
 * in `llm_statistics` under a pipeline, wrap it in an agent instead.
 *
 * Builds a [MusicPromptSettings] prompt (optional [systemPrompt] as SYSTEM framing — e.g.
 * "no lyrics, loopable" — and [description] as USER text), executes, aggregates, and returns the
 * media result. With a `BinaryDataStorage` in the chain the audio comes back as an
 * [ImagePromptExecutionResult.File] (the generic binary-media result, historically named "Image");
 * without one it degrades to [ImagePromptExecutionResult.Bytes].
 */
suspend fun PromptExecutor.generateMusic(
    llm: Llm,
    description: String,
    systemPrompt: String? = null,
): ImagePromptExecutionResult {
    // Single USER text — Lyria takes a free-text prompt and rejects system-instruction. The optional
    // [systemPrompt] framing ("instrumental, loopable, …") is prepended to the description, not sent
    // as a separate SYSTEM part.
    val text = if (!systemPrompt.isNullOrBlank()) "$systemPrompt\n\n$description" else description
    val prompt = Prompt(
        llm = llm,
        settings = MusicPromptSettings(),
        content = listOf(PromptContent(PromptContentType.USER, listOf(TextPromptPart(text)))),
    )
    val result = execute(prompt).aggregate()
    return result as? ImagePromptExecutionResult
        ?: throw PromptExecutorException(
            "Expected ImagePromptExecutionResult (media) for music generation, got ${result::class.simpleName}"
        )
}
