package org.nobirds.murmuration.prompt

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.llm.*

import org.nobirds.murmuration.internal.sha256

/**
 * Whole-prompt dump: PromptContent type plus the content of each PromptPart.
 * Text parts are printed as-is; binary parts are summarised as hash/size/mime to keep the
 * log readable. Used for DEBUG logging of what actually went to the LLM (after cache
 * resolution), without dumping the agent input.
 */
fun Prompt.dump(): String = buildString {
    appendLine("Prompt[model=${llm.model}, settings=${settings.dumpSettings()}]")
    appendContents(content)
}

fun List<PromptContent>.dump(label: String): String = buildString {
    appendLine("$label:")
    appendContents(this@dump)
}

fun PromptSettings?.dumpSettings(): String = when (this) {
    null -> "null"
    is TypedPromptSettings<*> -> "Typed(${type.simpleName})"
    is ToolsPromptSettings -> {
        val names = tools.flatMap { it.instruments }.map { it.name }
        "Tools(${names.size}: ${names.joinToString(", ")})"
    }
    else -> this::class.simpleName ?: "?"
}

private fun StringBuilder.appendContents(contents: List<PromptContent>) {
    if (contents.isEmpty()) {
        appendLine("  <empty>")
        return
    }
    contents.forEachIndexed { idx, c ->
        appendLine("  [#$idx ${c.type}] (parts=${c.parts.size})")
        c.parts.forEachIndexed { pIdx, part ->
            appendLine("    [$pIdx] ${part.dumpLine()}")
        }
    }
}

private fun PromptPart.dumpLine(): String = when (this) {
    is TextPromptPart -> "TEXT cache=${cacheStrategy.scope}\n${text.indentLines("      ")}"
    is DataPromptPart -> "DATA mime=$mimeType name=$name size=${data.size}b sha256=${data.sha256().take(16)} cache=${cacheStrategy.scope}"
    is FileDataPromptPart -> "FILE uri=$uri mime=$mimeType name=$name cache=${cacheStrategy.scope}"
    is ToolCallPromptPart -> "TOOL_CALL name=${call.name} args=${call.args} cache=${cacheStrategy.scope}"
    is ToolResultPromptPart -> "TOOL_RESULT name=$name cache=${cacheStrategy.scope}\n${result.toString().indentLines("      ")}"
}

private fun String.indentLines(indent: String): String =
    lineSequence().joinToString(separator = "\n") { "$indent$it" }
