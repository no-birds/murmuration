package org.nobirds.murmuration.prompt

import org.nobirds.murmuration.agent.PromptExecutorException
import org.nobirds.murmuration.llm.Llm

/**
 * Convenience: embed [text] into a dense vector through this executor directly — no agent layer,
 * so no [org.nobirds.murmuration.agent.ExecutionContext] is required. Builds an
 * [EmbeddingPromptSettings] prompt, executes, aggregates, and extracts the vector.
 *
 * Use this for embeddings outside a pipeline (e.g. a standalone service or a one-off job). When
 * the embedding happens inside a pipeline and should land in `llm_statistics` with retry, use the
 * `embeddingAgent` builder instead — it rides the agent layer.
 */
suspend fun PromptExecutor.embed(
    llm: Llm,
    text: String,
    outputDimensionality: Int = 768,
    taskType: String = "SEMANTIC_SIMILARITY",
): FloatArray {
    val prompt = Prompt(
        llm = llm,
        settings = EmbeddingPromptSettings(outputDimensionality, taskType),
        content = listOf(PromptContent(PromptContentType.USER, listOf(TextPromptPart(text)))),
    )
    val result = execute(prompt).aggregate()
    return (result as? EmbeddingPromptExecutionResult)?.embedding
        ?: throw PromptExecutorException(
            "Expected EmbeddingPromptExecutionResult, got ${result::class.simpleName}"
        )
}
