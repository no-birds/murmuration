package org.nobirds.murmuration.llm

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.prompt.*

interface Llm {
    val provider: String
    val family: String
    val name: String
    val version: String
    val stable: Boolean
    val model: String

    /**
     * Optional per-model pricing. [LlmCostCalculator] looks the [Llm] up by the actual model
     * name from the API (`PromptExecutionStatistics.modelName`) via [LlmRegistry] and reads
     * the profile from the matched constant.
     */
    val pricingProfile: LlmPricingProfile? get() = null

    /** Content modalities this model accepts on input (default text-only). See [requireModalities]. */
    val inputModalities: Set<Modality.Input> get() = setOf(Modality.Text)

    /** Content modalities this model can produce (default text-only). See [requireModalities]. */
    val outputModalities: Set<Modality.Output> get() = setOf(Modality.Text)

    /** Provider constraints / features this model has. Query via [capability]; default none. */
    val capabilities: Set<Capability> get() = emptySet()

    /**
     * Tests whether [actualModelName] (the model id from the API response) refers to this
     * constant. Default is a prefix match: Gemini/Vertex and similar providers return names
     * with a date-version suffix (e.g. `gemini-3-flash-preview-04-17` for the requested alias
     * `gemini-3-flash-preview`).
     *
     * Implementations may override (for example, to use an exact match) where the provider's
     * semantics require it. When several constants would match the same actual name,
     * [LlmRegistry] picks the one with the longest [model] string.
     */
    fun matches(actualModelName: String): Boolean =
        actualModelName.removePrefix("models/").startsWith(model.removePrefix("models/"))

    companion object {
        operator fun invoke(
            provider: String,
            family: String,
            name: String,
            version: String,
            stable: Boolean,
            model: String,
            pricingProfile: LlmPricingProfile? = null,
            inputModalities: Set<Modality.Input> = setOf(Modality.Text),
            outputModalities: Set<Modality.Output> = setOf(Modality.Text),
            capabilities: Set<Capability> = emptySet(),
        ): Llm = LlmImpl(provider, family, name, version, stable, model, pricingProfile, inputModalities, outputModalities, capabilities)
    }

    private data class LlmImpl(
        override val provider: String,
        override val family: String,
        override val name: String,
        override val version: String,
        override val stable: Boolean,
        override val model: String,
        override val pricingProfile: LlmPricingProfile?,
        override val inputModalities: Set<Modality.Input>,
        override val outputModalities: Set<Modality.Output>,
        override val capabilities: Set<Capability>,
    ) : Llm
}
