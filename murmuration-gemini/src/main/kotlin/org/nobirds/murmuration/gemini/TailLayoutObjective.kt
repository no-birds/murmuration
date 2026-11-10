package org.nobirds.murmuration.gemini

import com.google.genai.types.FunctionDeclaration
import com.google.genai.types.Schema
import com.google.genai.types.Type
import org.nobirds.murmuration.gemini.GeminiToolDeclarations.collectSignatures
import org.nobirds.murmuration.gemini.fst.FstStateModel

/**
 * States a single field costs alone: N of a one-tool declaration with this field minus N of an empty tool.
 * That is what one merge of equal fields saves. Cached by signature (the signature fixes the shape).
 * Port of `layout.py:weight`.
 */
internal class FieldWeights {
    private val cache = HashMap<String, Int>()
    private val emptyStates by lazy { FstStateModel.states(listOf(FunctionDeclaration.builder().name("t").build())) }

    fun of(signature: String, key: String, value: Schema): Int = cache.getOrPut(signature) {
        val parameters = Schema.builder().type(Type.Known.OBJECT).properties(mapOf(key to value)).required(listOf(key)).build()
        val one = FunctionDeclaration.builder().name("t").parameters(parameters).build()
        FstStateModel.states(listOf(one)) - emptyStates
    }
}

/**
 * Myhill-Nerode proxy of the FST size for one profile (port of `layout.py:_fields` and `proxy`): the sum, over
 * distinct pairs (right context, suffix of an object from a field), of the field weight plus [CODE_LENGTH] for a
 * shifted field. Built on canonical schemas with the path rule already applied: a field blocked by the rule
 * stays canonical, never matches a rank entry, and merges only with equal blocked fields.
 */
internal class ProfileObjective private constructor(
    private val objects: List<ObjectNode>,
    /** Signature -> number of fields with it, in first-seen order; blocked fields are not counted. */
    val frequency: Map<String, Int>,
) {
    internal class Field(val signature: String, val weight: Int)

    /** An object with properties; [parent] is the index of the enclosing object or `-1`. */
    internal class ObjectNode(
        val parent: Int,
        val parentSignature: String?,
        val marker: String,
        val fields: LinkedHashMap<String, Field>,
    )

    private data class ContextKey(val parent: Int, val parentTail: Int, val marker: String, val item: String?)

    /** Proxy cost when the signatures in [rank] (signature -> position) are moved to the tail in that order. */
    fun cost(rank: Map<String, Int>): Int {
        val suffixIds = HashMap<Pair<String, Int>, Int>()
        val contextIds = HashMap<ContextKey, Int>()
        val seen = HashSet<Long>()
        val orderedSignatures = arrayOfNulls<List<String>>(objects.size)
        val suffixesOf = arrayOfNulls<IntArray>(objects.size)
        val contextOf = IntArray(objects.size)
        var cost = 0
        objects.forEachIndexed { index, node ->
            val keys = node.fields.keys.sortedWith { a, b -> compareKeys(rank, node, a, b) }
            val signatures = keys.map { node.fields.getValue(it).signature }
            val suffixes = IntArray(signatures.size + 1)
            for (j in signatures.indices.reversed()) {
                suffixes[j] = suffixIds.getOrPut(signatures[j] to suffixes[j + 1]) { suffixIds.size + 1 }
            }
            val context = if (node.parent < 0) 0 else {
                val parentSignatures = checkNotNull(orderedSignatures[node.parent])
                val at = parentSignatures.indexOf(node.parentSignature)
                check(at >= 0) { "parent field '${node.parentSignature}' is not among the parent's signatures" }
                val item = if ("[]" in node.marker) node.parentSignature else null
                val key = ContextKey(contextOf[node.parent], checkNotNull(suffixesOf[node.parent])[at + 1], node.marker, item)
                contextIds.getOrPut(key) { contextIds.size + 1 }
            }
            orderedSignatures[index] = signatures
            suffixesOf[index] = suffixes
            contextOf[index] = context
            keys.forEachIndexed { j, key ->
                if (seen.add((context.toLong() shl 32) or suffixes[j].toLong())) {
                    val field = node.fields.getValue(key)
                    cost += field.weight + if (field.signature in rank) CODE_LENGTH else 0
                }
            }
        }
        return cost
    }

    /** Python `sorted(key=(1, rank) if shifted else (0, key))`: untouched fields by key first, then the tail by rank. */
    private fun compareKeys(rank: Map<String, Int>, node: ObjectNode, a: String, b: String): Int {
        val ra = rank[node.fields.getValue(a).signature]
        val rb = rank[node.fields.getValue(b).signature]
        return when {
            ra == null && rb == null -> a.compareTo(b)
            ra == null -> -1
            rb == null -> 1
            else -> ra.compareTo(rb)
        }
    }

    companion object {
        /** Length of the tail prefix: `~` + one code character. */
        const val CODE_LENGTH = 2

        /** Objective of the tools in [declarations]; [weights] is shared across profiles. */
        fun of(declarations: List<FunctionDeclaration>, weights: FieldWeights): ProfileObjective {
            val objects = ArrayList<ObjectNode>()
            val frequency = LinkedHashMap<String, Int>()
            for (declaration in declarations) {
                val root = declaration.parameters().orElse(null) ?: continue
                val signatures = LinkedHashMap<List<String>, MutableSet<String>>()
                root.collectSignatures(emptyList(), signatures)
                Walker(weights, signatures, objects, frequency).walk(root, emptyList(), -1, null, "")
            }
            return ProfileObjective(objects, frequency)
        }
    }

    private class Walker(
        private val weights: FieldWeights,
        private val pathSignatures: Map<List<String>, Set<String>>,
        private val objects: MutableList<ObjectNode>,
        private val frequency: MutableMap<String, Int>,
    ) {
        fun walk(schema: Schema, path: List<String>, parent: Int, parentSignature: String?, marker: String) {
            val properties = schema.properties().orElse(emptyMap())
            if (schema.type().orElse(null)?.knownEnum() == Type.Known.OBJECT && properties.isNotEmpty()) {
                val me = objects.size
                val fields = LinkedHashMap<String, Field>()
                objects += ObjectNode(parent, parentSignature, marker, fields)
                val signatures = properties.keys.associateWith { FieldSignature.of(schema, it) }
                val blocked = properties.keys.filterTo(HashSet()) { (pathSignatures[path + it]?.size ?: 0) > 1 }
                properties.forEach { (key, value) ->
                    val signature = signatures.getValue(key)
                    val weight = if (value.hasObject()) key.length + 2 else weights.of(signature, key, value)
                    fields[key] = Field(if (key in blocked) "$signature@canon" else signature, weight)
                    if (key !in blocked) frequency.merge(signature, 1, Int::plus)
                }
                properties.forEach { (key, value) -> walk(value, path + key, me, signatures.getValue(key), "") }
                return
            }
            schema.items().ifPresent { walk(it, path, parent, parentSignature, marker + "[]") }
            schema.anyOf().ifPresent { variants ->
                variants.forEachIndexed { i, variant -> walk(variant, path, parent, parentSignature, "$marker|$i") }
            }
        }

        private fun Schema.hasObject(): Boolean =
            type().orElse(null)?.knownEnum() == Type.Known.OBJECT ||
                items().map { it.hasObject() }.orElse(false) ||
                anyOf().map { variants -> variants.any { it.hasObject() } }.orElse(false)
    }
}
