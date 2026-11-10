package org.nobirds.murmuration.prompt

import org.nobirds.murmuration.llm.cacheKind

/**
 * Drops earlier conversation chunks whose every marked part has been **superseded** by a
 * later chunk with the same `replaceableSectionId`. A chunk is either:
 *  - a standalone content (no tool exchange), or
 *  - a paired `MODEL_with_tool_calls` + next `TOOL_with_tool_results` content (atomic for
 *    provider-side validation: dropping half breaks the chat structure).
 *
 * **Rules**:
 *  - Chunk with no marked parts is left intact.
 *  - Chunk with marked parts is kept only when, for every id present anywhere in it, this
 *    chunk is the latest occurrence of that id across the whole conversation.
 *  - Otherwise the whole chunk (both contents of a pair, or the single standalone) is dropped.
 *
 * **Why chunks, not contents**: tool-call and tool-result parts come paired. Dropping a
 * single tool_call without its result (or vice versa) breaks provider-side validation
 * (Gemini, OpenAI). Treating the pair as the unit of replacement keeps the model/tool
 * pairing intact — either both go or both stay.
 *
 * Use cases:
 *  - `replaceableSectionId="current-output-snapshot"` on snapshot text → only the latest
 *    snapshot survives.
 *  - `replaceableSectionId="phase-description"` on phase instructions/tools blocks → only
 *    the current phase description survives; old WORKING text vanishes after SELF_REVIEW
 *    is appended.
 *  - `replaceableSectionId="tool-exchange-history"` on every model+tool exchange → only
 *    the freshest round-trip exchange remains; everything earlier is dropped (its effect is
 *    already reflected in the latest snapshot).
 */
fun Prompt.pruneReplaceableSections(): Prompt {
    // Step 1: group contents into chunks. A MODEL message carrying any ToolCall part is
    // paired with the immediately-following TOOL message (which carries the matching
    // ToolResult parts). Anything else is a standalone chunk.
    val chunks = mutableListOf<IntRange>()
    var i = 0
    while (i < content.size) {
        val c = content[i]
        val isPaired = c.type == PromptContentType.MODEL
            && c.parts.any { it is ToolCallPromptPart }
            && i + 1 < content.size
            && content[i + 1].type == PromptContentType.TOOL
        if (isPaired) {
            chunks += (i..i + 1)
            i += 2
        } else {
            chunks += (i..i)
            i++
        }
    }

    // Step 2: collect last-chunk index per section id.
    val lastChunkForId = mutableMapOf<String, Int>()
    chunks.forEachIndexed { chunkIdx, range ->
        for (idx in range) {
            content[idx].parts.forEach { part ->
                val id = part.replaceableSectionId() ?: return@forEach
                lastChunkForId[id] = chunkIdx
            }
        }
    }
    if (lastChunkForId.isEmpty()) return this

    // Step 3: keep chunks whose every id is at this chunk's index.
    val keptContents = mutableListOf<PromptContent>()
    chunks.forEachIndexed { chunkIdx, range ->
        val ids = mutableSetOf<String>()
        for (idx in range) {
            content[idx].parts.forEach { part ->
                part.replaceableSectionId()?.let { ids += it }
            }
        }
        val keep = ids.isEmpty() || ids.all { lastChunkForId[it] == chunkIdx }
        if (keep) {
            for (idx in range) keptContents += content[idx]
        }
    }

    if (keptContents.size == content.size) return this
    return Prompt(llm, settings, keptContents)
}

/**
 * Executor decorator applying [pruneReplaceableSections] to every prompt before delegating.
 * When [enabled] is `false`, returns the original executor unchanged (identity, no wrapping).
 * Prompts of a model whose [cacheKind] is [CacheKind.MARKS] pass through unpruned even when enabled.
 *
 * Pair this with a startup-time warning when explicit caching is disabled — pruning replaces
 * the implicit-cache prefix on every round-trip, defeating implicit caching. The decorator
 * itself stays silent so each prompt costs zero extra allocations beyond the prune itself.
 */
fun PromptExecutor.withReplaceableSectionPruning(enabled: Boolean): PromptExecutor =
    if (!enabled) this
    else PromptExecutor { prompt ->
        // A MARKS-kind model's history is append-only (the agent marks its tail): never pruned, whatever the flag.
        execute(if (prompt.llm.cacheKind == CacheKind.MARKS) prompt else prompt.pruneReplaceableSections())
    }
