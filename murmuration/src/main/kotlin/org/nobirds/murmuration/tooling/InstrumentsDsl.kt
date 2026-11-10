package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.prompt.InstrumentDeclaration
import org.nobirds.murmuration.prompt.InstrumentParameter
import kotlinx.serialization.json.JsonElement
import kotlin.reflect.KType

/**
 * Build a custom list of [InstrumentDeclaration]s by pulling tools out of several source
 * objects, optionally tagging each group with `@Phases`. Intended for agents that need
 * to compose a phase-driven tool surface from multiple existing builders without
 * copy-pasting `@Instrument`-annotated methods.
 *
 * Example:
 * ```
 * val tools = buildInstruments {
 *     from(planner) { exclude("transitionTo", "addTask"); inPhases(MyState.PLANNING) }
 *     from(actions) { exclude("transitionTo"); inPhases(MyState.ACTIONS) }
 *     from(memory) { include("addMemory", "updateMemory") }   // whitelist: exactly these
 *     from(this@AgentBuilder)   // no phases, no filter — available everywhere
 * }
 * ```
 *
 * `@Phases` is attached at runtime via [withAdditionalAnnotations]; downstream
 * `InstrumentDeclaration.phases` (the extension in [Phases]) picks it up the same way
 * it picks up a compile-time annotation on the source method.
 *
 * Filtering is scoped per `from(...)` group and comes in two symmetric forms:
 * [FromGroupBuilder.exclude] (drop the named ones) and [FromGroupBuilder.include] (keep
 * exactly the named ones). A global filter was tempting but error-prone: it would also
 * strip e.g. `transitionTo` from a `from(self)` call, which is almost never what the
 * caller wants.
 */
fun buildInstruments(block: InstrumentSetBuilder.() -> Unit): List<InstrumentDeclaration<*>> =
    InstrumentSetBuilder().apply(block).build()

class InstrumentSetBuilder internal constructor() {
    private val factory = InstrumentFactory()
    private val collected = mutableListOf<InstrumentDeclaration<*>>()

    /**
     * Pull `@Instrument`-annotated tools from [target], optionally narrow the group by name
     * ([FromGroupBuilder.exclude] / [FromGroupBuilder.include]) and tag survivors with
     * `@Phases` via the configuration block. Filtering is per-group on purpose — a global
     * one would also strip e.g. `transitionTo` from a `from(self)` call, which is almost
     * never what you want.
     *
     * @throws IllegalArgumentException if the group's filter is inconsistent with the donor
     *   (both filters at once, or `include` naming a tool the donor does not declare) —
     *   building a tool surface out of stale names fails loudly, see [FromGroupBuilder.select].
     */
    fun from(target: Any, configure: FromGroupBuilder.() -> Unit = {}) {
        val group = FromGroupBuilder().apply(configure)
        val raw = factory.create(target).instruments
        collected.addAll(group.select(raw, target).withPhases(group.phaseNames))
    }

    /**
     * Final dedup pass: Gemini rejects requests with duplicate function declarations
     * (`400 Duplicate function declaration found: <name>`). Multi-source compositions
     * naturally produce duplicates (e.g. classic `setGameStatus` historically lived on
     * both Planner and Synthesis builders). First-occurrence wins — caller is expected
     * to order `from(...)` groups so the canonical implementation comes first.
     *
     * NB: when duplicates have DIFFERENT phase sets (PLANNING vs SYNTHESIS), this pass
     * silently loses one of them. The cleanest fix is unifying the source builders so
     * the tool exists in exactly one place.
     * Until then, callers must dedupe at the source.
     */
    internal fun build(): List<InstrumentDeclaration<*>> = collected.distinctBy { it.name }
}

class FromGroupBuilder internal constructor() {
    internal var phaseNames: List<String> = emptyList()
        private set
    internal var excludedNames: Set<String> = emptySet()
        private set
    internal var includedNames: Set<String>? = null
        private set

    /** Tag this group's tools with `@Phases(states.map { it.name })`. */
    fun inPhases(vararg states: Enum<*>) {
        phaseNames = states.map { it.name }
    }

    /** Drop tools whose `name` is in [names] from this group. */
    fun exclude(vararg names: String) {
        excludedNames = names.toSet()
    }

    /**
     * WHITELIST: keep EXACTLY the tools named in [names] from this group's donor and drop
     * everything else. Symmetric counterpart of [exclude], for the common "narrow agent that
     * mounts three tools of a fat donor" case — expressing such a surface by subtraction says
     * nothing about intent and silently widens whenever the donor grows a tool.
     *
     * A name the donor does not declare fails [buildInstruments] with `IllegalArgumentException`,
     * never a silent skip: a whitelist is a claim about the donor, and a stale claim means the agent
     * quietly lost a tool it believes it has. Note this fires where `buildInstruments` is CALLED —
     * with a lazily built `instruments` list that is runtime, not compile time, so a pin test on the
     * resulting tool set is the only early signal. The two filters are mutually exclusive within one
     * group — pick the one that states the intent.
     *
     * An EMPTY whitelist is rejected for the same reason: `include()` without names would hand the
     * agent an empty schema — a tool-less agent that silently does nothing, which no caller means.
     */
    fun include(vararg names: String) {
        require(names.isNotEmpty()) {
            "buildInstruments: include()/only() called with no names — an empty whitelist would leave " +
                "the agent with an empty schema. Name the tools to keep, or drop the include(...) call " +
                "to keep the donor's full surface."
        }
        includedNames = names.toSet()
    }

    /** Alias of [include] that reads better at the call site: `only("addMemory", "updateMemory")`. */
    fun only(vararg names: String) = include(*names)

    /**
     * Apply this group's filter to the donor's [raw] declarations. Include wins nothing by
     * itself — the two filters cannot be combined, see [include].
     */
    internal fun select(raw: List<InstrumentDeclaration<*>>, target: Any): List<InstrumentDeclaration<*>> {
        val kept = includedNames ?: return if (excludedNames.isEmpty()) raw else raw.filter { it.name !in excludedNames }
        val donor = target::class.simpleName ?: target::class.toString()
        require(excludedNames.isEmpty()) {
            "buildInstruments: from($donor) uses both include(...) and exclude(...) — pick one: " +
                "include=${kept.sorted()}, exclude=${excludedNames.sorted()}"
        }
        val declared = raw.mapTo(mutableSetOf()) { it.name }
        val unknown = kept - declared
        require(unknown.isEmpty()) {
            "buildInstruments: from($donor) include(...) names tools the donor does not declare: " +
                "${unknown.sorted()}. Declared: ${declared.sorted()}"
        }
        return raw.filter { it.name in kept }
    }
}

/**
 * Wrap a declaration so that its `annotations` list contains the original ones plus
 * [extra]. All other fields (`name`, `description`, `parameters`, `returnType`, `obj`,
 * `invoke`) are delegated to the underlying declaration unchanged.
 */
fun <R> InstrumentDeclaration<R>.withAdditionalAnnotations(
    extra: List<Annotation>,
): InstrumentDeclaration<R> = AnnotatedInstrumentDeclaration(this, extra)

private fun List<InstrumentDeclaration<*>>.withPhases(
    phaseNames: List<String>,
): List<InstrumentDeclaration<*>> {
    if (phaseNames.isEmpty()) return this
    val anno = Phases(phases = phaseNames.toTypedArray())
    return map { it.withAdditionalAnnotations(listOf(anno)) }
}

private class AnnotatedInstrumentDeclaration<R>(
    private val delegate: InstrumentDeclaration<R>,
    private val extra: List<Annotation>,
) : InstrumentDeclaration<R> {
    override val name: String get() = delegate.name
    override val description: String get() = delegate.description
    override val parameters: List<InstrumentParameter> get() = delegate.parameters
    override val returnType: KType get() = delegate.returnType
    override val annotations: List<Annotation> get() = delegate.annotations + extra
    override val obj: Any get() = delegate.obj
    override val instruments: List<InstrumentDeclaration<*>> get() = listOf(this)
    override fun prepareWireArguments(args: Map<String, JsonElement>): Map<String, Any?> =
        delegate.prepareWireArguments(args)
    override suspend fun invoke(args: Map<String, Any?>): R = delegate.invoke(args)
}
