package org.nobirds.murmuration.llm

import org.nobirds.murmuration.prompt.DataPromptPart
import org.nobirds.murmuration.prompt.FileDataPromptPart
import org.nobirds.murmuration.prompt.Prompt
import kotlin.reflect.KClass

/**
 * A content modality on a model's input or output axis (see [Llm.inputModalities] /
 * [Llm.outputModalities]). A model lists what it actually does as a set — an image model outputs
 * `{Text, Image}`, an embedder `{Embedding(768)}`, Lyria `{Audio}` — rather than declaring a marker
 * interface per modality.
 *
 * Split into [Input] / [Output] so the axes can't be confused: `Text/Image/Audio/Video` are valid on
 * both (they implement both, no duplication), while [Thoughts] and [Embedding] are output-only — the
 * type system then rejects e.g. `inputModalities = setOf(Thoughts)`. Sealed so a modality can carry
 * data ([Embedding.dimensions]); membership is therefore checked by type, not by value.
 */
sealed interface Modality {
    sealed interface Input : Modality
    sealed interface Output : Modality

    data object Text : Input, Output
    data object Image : Input, Output
    data object Audio : Input, Output
    data object Video : Input

    /** The reasoning trace on the output axis (`includeThoughts`) — distinct from the [Thinking] capability. */
    data object Thoughts : Output

    /** Dense-vector output; [dimensions] is the model's native embedding size (e.g. 768). */
    data class Embedding(val dimensions: Int) : Output
}

/**
 * Maps a binary part's MIME type to the input-modality type it carries, or `null` when it implies no
 * media-modality constraint (text, documents, unknown types — not validated here, to avoid breaking
 * legitimate inputs the mapping doesn't model).
 */
private fun mimeModality(mime: String?): KClass<out Modality.Input>? = when {
    mime == null -> null
    mime.startsWith("image/") -> Modality.Image::class
    mime.startsWith("audio/") -> Modality.Audio::class
    mime.startsWith("video/") -> Modality.Video::class
    else -> null
}

/**
 * Fail-fast guard wiring [Llm.inputModalities] / [Llm.outputModalities] to actual usage: an agent
 * declares the [requiredOutput] modality it needs (`Modality.Image::class` for an image agent,
 * `Modality.Audio::class` for music, `Modality.Text::class` for text, `Modality.Embedding::class` for
 * embeddings), and every media input part must be one the model accepts. A mismatch is a wiring bug —
 * the wrong model for the agent, or a model that can't take this input — so throw rather than let the
 * provider reject it with an opaque 400. Membership is by type (a modality may carry data).
 */
fun Prompt.requireModalities(requiredOutput: KClass<out Modality.Output>) {
    require(llm.outputModalities.any { requiredOutput.isInstance(it) }) {
        "Model '${llm.model}' cannot produce ${requiredOutput.simpleName} output (declares ${llm.outputModalities}) — wrong model for this agent."
    }
    content.asSequence()
        .flatMap { it.parts }
        .mapNotNull { part ->
            when (part) {
                is DataPromptPart -> mimeModality(part.mimeType)
                is FileDataPromptPart -> mimeModality(part.mimeType)
                else -> null
            }
        }
        .forEach { input ->
            require(llm.inputModalities.any { input.isInstance(it) }) {
                "Model '${llm.model}' cannot accept ${input.simpleName} input (declares ${llm.inputModalities})."
            }
        }
}
