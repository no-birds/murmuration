package org.nobirds.murmuration.gemini

import com.google.genai.types.FunctionDeclaration
import com.google.genai.types.Schema
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.nobirds.murmuration.gemini.GeminiToolDeclarations.rewrite
import org.nobirds.murmuration.gemini.fst.FstStateModel

/** Dumped tool profiles (`fst/profiles`), the frozen reference table and the Python reference numbers. */
internal object TailLayoutTestSupport {

    /** The 28 dumped profiles by name: declarations as they are before any layout. */
    val profiles: Map<String, List<FunctionDeclaration>> by lazy {
        val names = (Json.parseToJsonElement(resource("fst/layout/expected.json")) as JsonObject).keys
        names.sorted().associateWith { name ->
            (Json.parseToJsonElement(resource("fst/profiles/$name.json")) as JsonArray)
                .map { FunctionDeclaration.fromJson(it.jsonObject.toString()) }
        }
    }

    /** Reference N per profile with the frozen table applied (`union.py` output). */
    val expectedStates: Map<String, Int> by lazy {
        (Json.parseToJsonElement(resource("fst/layout/expected.json")) as JsonObject)
            .mapValues { it.value.jsonPrimitive.int }
    }

    /** Frozen table rows: code, canonical key, signature hash (`wire-keys.tsv`), in tail order. */
    val frozenRows: List<Triple<Char, String, String>> by lazy {
        resource("fst/layout/wire-keys.tsv").lines().filter { it.isNotBlank() }.map { line ->
            val (code, key, hash) = line.split('\t')
            Triple(code.single(), key, hash)
        }
    }

    /** Every distinct field signature of [declarations], over all objects at every level. */
    fun signaturesOf(declarations: List<FunctionDeclaration>): Set<String> {
        val out = linkedSetOf<String>()
        fun walk(schema: Schema) {
            schema.properties().orElse(emptyMap()).forEach { (key, value) ->
                out += FieldSignature.of(schema, key)
                walk(value)
            }
            schema.items().ifPresent { walk(it) }
            schema.anyOf().ifPresent { variants -> variants.forEach { walk(it) } }
        }
        declarations.forEach { d -> d.parameters().ifPresent { walk(it) } }
        return out
    }

    /** Vertex FST size N of [declarations] after [layout] has been applied to every tool. */
    fun states(declarations: List<FunctionDeclaration>, layout: GeminiWireKeyLayout): Int =
        FstStateModel.states(declarations.map { declaration ->
            val root = declaration.parameters().orElse(null) ?: return@map declaration
            val map = GeminiToolDeclarations.wireKeyMapOf(root, layout)
            if (map.isEmpty) declaration else declaration.toBuilder().parameters(root.rewrite(emptyList(), map)).build()
        })

    fun resource(path: String): String =
        checkNotNull(javaClass.classLoader.getResourceAsStream(path)) { "no resource $path" }
            .use { it.readBytes().toString(Charsets.UTF_8) }
}
