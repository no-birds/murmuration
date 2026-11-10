package org.nobirds.murmuration.gemini

import com.google.genai.types.FunctionDeclaration
import com.google.genai.types.Schema
import com.google.genai.types.Tool
import org.nobirds.murmuration.prompt.InstrumentDeclaration
import org.nobirds.murmuration.schema.toFunctionSchema

/**
 * What one request or cache create sends for its tools, built once from a single `layout`:
 * the wire [tool] (`null` when there are no instruments) and the per-tool [wireKeyMaps] that
 * the history converter uses, so declarations and history can never come from different layouts.
 */
class WireDeclarations(
    val declarations: List<FunctionDeclaration>,
    val wireKeyMaps: Map<String, WireKeyMap>,
) {
    /** A single [Tool] carrying all [declarations]; `null` when there are none. */
    val tool: Tool? = if (declarations.isEmpty()) null else Tool.builder().functionDeclarations(declarations).build()

    /** History converter over the same wire key maps as [tool]. */
    val historyConverter: GeminiHistoryConverter = GeminiHistoryConverter(wireKeyMaps)
}

/**
 * The single point where mounted instruments become Gemini function declarations.
 *
 * Both the generate-content request and the cached-content create call go through here, so what
 * is measured (size ratchets, FST probes) is exactly what reaches the provider. Every call names its
 * [GeminiWireKeyLayout] explicitly; [GeminiWireKeyLayout.NONE] is the identity.
 */
object GeminiToolDeclarations {

    /**
     * Declarations for [instruments] laid out by [layout], in the given order, with the wire key map of
     * every tool. Group rule for `anyOf`: a key gets a wire name only if all variants that declare it at
     * the same position have the same signature; otherwise it stays canonical in every variant.
     */
    fun build(instruments: List<InstrumentDeclaration<*>>, layout: GeminiWireKeyLayout): WireDeclarations {
        val maps = LinkedHashMap<String, WireKeyMap>()
        val declarations = instruments.map { instrument ->
            val schema = instrument.toFunctionSchema()
            val canonical = schema.parameters.toGeminiSchema()
            val keyMap = wireKeyMapOf(canonical, layout)
            val wire = if (keyMap.isEmpty) canonical else canonical.rewrite(emptyList(), keyMap)
            if (!keyMap.isEmpty) maps[schema.name] = keyMap
            FunctionDeclaration.builder()
                .name(schema.name)
                .description(schema.description)
                .parameters(wire)
                .build()
        }
        return WireDeclarations(declarations, maps)
    }

    /** One [FunctionDeclaration] per instrument, in the given order. */
    fun of(instruments: List<InstrumentDeclaration<*>>, layout: GeminiWireKeyLayout): List<FunctionDeclaration> =
        build(instruments, layout).declarations

    /** A single [Tool] carrying all declarations of [instruments]; `null` when there are none. */
    fun toolOf(instruments: List<InstrumentDeclaration<*>>, layout: GeminiWireKeyLayout): Tool? =
        build(instruments, layout).tool

    internal fun wireKeyMapOf(root: Schema, layout: GeminiWireKeyLayout): WireKeyMap {
        val signatures = LinkedHashMap<List<String>, MutableSet<String>>()
        root.collectSignatures(emptyList(), signatures)
        val wireNames = LinkedHashMap<List<String>, String>()
        signatures.forEach { (path, sigs) ->
            require(!path.last().startsWith(WIRE_KEY_MARKER)) { "canonical key '${path.last()}' must not start with '$WIRE_KEY_MARKER'" }
            if (sigs.size == 1) layout.wireKeyOf(sigs.single(), path.last())?.let { wireNames[path] = it }
        }
        val map = WireKeyMap(wireNames)
        root.requireNoCollisions(emptyList(), map)
        return map
    }

    internal fun Schema.collectSignatures(path: List<String>, out: MutableMap<List<String>, MutableSet<String>>) {
        properties().orElse(emptyMap()).forEach { (key, value) ->
            out.getOrPut(path + key) { linkedSetOf() }.add(FieldSignature.of(this, key))
            value.collectSignatures(path + key, out)
        }
        items().ifPresent { it.collectSignatures(path, out) }
        anyOf().ifPresent { variants -> variants.forEach { it.collectSignatures(path, out) } }
    }

    private fun Schema.requireNoCollisions(path: List<String>, map: WireKeyMap) {
        val properties = properties().orElse(emptyMap())
        val names = properties.keys.map { map.wireNameAt(path + it) ?: it }
        require(names.toSet().size == names.size) { "wire key collision under ${path.joinToString(".")}: $names" }
        properties.forEach { (key, value) -> value.requireNoCollisions(path + key, map) }
        items().ifPresent { it.requireNoCollisions(path, map) }
        anyOf().ifPresent { variants -> variants.forEach { it.requireNoCollisions(path, map) } }
    }

    internal fun Schema.rewrite(path: List<String>, map: WireKeyMap): Schema {
        val builder = toBuilder()
        properties().ifPresent { properties ->
            builder.properties(properties.entries.associate { (key, value) ->
                (map.wireNameAt(path + key) ?: key) to value.rewrite(path + key, map)
            })
        }
        required().ifPresent { required -> builder.required(required.map { map.wireNameAt(path + it) ?: it }) }
        items().ifPresent { builder.items(it.rewrite(path, map)) }
        anyOf().ifPresent { variants -> builder.anyOf(variants.map { it.rewrite(path, map) }) }
        return builder.build()
    }
}
