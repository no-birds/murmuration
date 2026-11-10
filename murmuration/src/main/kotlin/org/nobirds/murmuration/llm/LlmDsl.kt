package org.nobirds.murmuration.llm

import org.nobirds.murmuration.prompt.ThinkingLevel

/**
 * Declarative builder for [Llm] constants. Reads the three descriptor axes — input/output
 * modalities and capabilities — as nested blocks instead of nested `setOf(...)` arguments:
 *
 * ```
 * val VERTEX_GEMINI_3_FLASH = llm("vertex", "gemini-3-flash-preview") {
 *     name = "Gemini 3 Flash"; version = "3"
 *     pricing = GEMINI_3_FLASH_PRICING
 *     input  { text(); image(); audio(); video() }
 *     output { text(); thoughts() }
 *     caching(minTokens = 4096)
 *     thinking { level() }
 *     toolCalling()
 * }
 * ```
 *
 * Modalities default to text-only ([Modality.Text]) when the block is omitted, matching [Llm].
 */
fun llm(provider: String, model: String, block: LlmBuilder.() -> Unit): Llm =
    LlmBuilder(provider, model).apply(block).build()

@DslMarker
annotation class LlmDslMarker

@LlmDslMarker
class LlmBuilder(private val provider: String, private val model: String) {
    var family: String = ""
    var name: String = model
    var version: String = ""
    var stable: Boolean = false
    var pricing: LlmPricingProfile? = null

    private var inputs: Set<Modality.Input> = setOf(Modality.Text)
    private var outputs: Set<Modality.Output> = setOf(Modality.Text)
    private val capabilities = mutableSetOf<Capability>()

    fun input(block: InputModalitiesBuilder.() -> Unit) {
        inputs = InputModalitiesBuilder().apply(block).build()
    }

    fun output(block: OutputModalitiesBuilder.() -> Unit) {
        outputs = OutputModalitiesBuilder().apply(block).build()
    }

    fun caching(minTokens: Int) { capabilities += Caching(minTokens) }
    fun thinking(block: ThinkingModesBuilder.() -> Unit) { capabilities += Thinking(ThinkingModesBuilder().apply(block).build()) }
    fun toolCalling() { capabilities += ToolCalling }

    internal fun build(): Llm = Llm(
        provider = provider,
        family = family,
        name = name,
        version = version,
        stable = stable,
        model = model,
        pricingProfile = pricing,
        inputModalities = inputs,
        outputModalities = outputs,
        capabilities = capabilities.toSet(),
    )
}

@LlmDslMarker
class InputModalitiesBuilder {
    private val set = mutableSetOf<Modality.Input>()
    fun text() { set += Modality.Text }
    fun image() { set += Modality.Image }
    fun audio() { set += Modality.Audio }
    fun video() { set += Modality.Video }
    internal fun build(): Set<Modality.Input> = set.toSet()
}

@LlmDslMarker
class OutputModalitiesBuilder {
    private val set = mutableSetOf<Modality.Output>()
    fun text() { set += Modality.Text }
    fun image() { set += Modality.Image }
    fun audio() { set += Modality.Audio }
    fun thoughts() { set += Modality.Thoughts }
    fun embedding(dimensions: Int) { set += Modality.Embedding(dimensions) }
    internal fun build(): Set<Modality.Output> = set.toSet()
}

@LlmDslMarker
class ThinkingModesBuilder {
    private val set = mutableSetOf<ThinkingMode>()
    /** Tier-controlled thinking. Pass [supported] tiers when the provider rejects some of them (400); no args = all four accepted. */
    fun level(vararg supported: ThinkingLevel) { set += ThinkingMode.Level(supported.toSet().takeIf { it.isNotEmpty() }) }
    fun budget(maxBudget: Int? = null) { set += ThinkingMode.Budget(maxBudget) }
    internal fun build(): Set<ThinkingMode> = set.toSet()
}
