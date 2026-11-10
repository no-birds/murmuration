package org.nobirds.murmuration.gemini

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.gemini.TailLayoutTestSupport.frozenRows
import org.nobirds.murmuration.gemini.TailLayoutTestSupport.profiles
import org.nobirds.murmuration.gemini.TailLayoutTestSupport.states
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The Kotlin signatures and the planner against the Python reference
 * (`layout.py`, `union.py`, `sigs.py`).
 */
class TailLayoutPlannerTest {

    @Test
    fun `field signatures and hashes equal the Python ones on every profile`() {
        val golden = Json.parseToJsonElement(TailLayoutTestSupport.resource("fst/layout/signatures.json")) as JsonObject
        assertEquals(golden.keys, profiles.keys)
        val mismatches = golden.entries.mapNotNull { (name, entry) ->
            val hashes = TailLayoutTestSupport.signaturesOf(profiles.getValue(name)).map { FieldSignature.hash(it) }.toSortedSet()
            val expected = entry as JsonObject
            val digest = FieldSignature.hash(hashes.joinToString("\n"))
            if (hashes.size == expected.getValue("count").jsonPrimitive.int && digest == expected.getValue("digest").jsonPrimitive.content) null
            else "$name: kotlin ${hashes.size}/$digest, python ${expected["count"]}/${expected["digest"]}"
        }
        assertEquals(emptyList<String>(), mismatches)
    }

    @Test
    fun `frozen table through the layout gives exactly the reference N on every profile`() {
        val bySignature = profiles.values.flatMap { TailLayoutTestSupport.signaturesOf(it) }.associateBy { FieldSignature.hash(it) }
        val layout = GeminiWireKeyLayout.ofSignatures(frozenRows.associate { (code, key, hash) ->
            val signature = checkNotNull(bySignature[hash]) { "signature $hash ($key) is absent from the profiles" }
            assertTrue(signature.startsWith(key), "$key vs $signature")
            signature to code
        })
        val mismatches = TailLayoutTestSupport.expectedStates.mapNotNull { (name, n) ->
            val actual = states(profiles.getValue(name), layout)
            if (actual == n) null else "$name: kotlin=$actual python=$n"
        }
        assertEquals(emptyList<String>(), mismatches)
    }

    @Test
    fun `planner from scratch stays within 50 of the frozen table on every profile`() {
        val t0 = System.nanoTime()
        val order = TailLayoutPlanner(profiles).plan()
        val planned = GeminiWireKeyLayout.ofSignatures(TailLayoutPlanner.codesOf(order))
        val bySignature = profiles.values.flatMap { TailLayoutTestSupport.signaturesOf(it) }.associateBy { FieldSignature.hash(it) }
        val frozen = GeminiWireKeyLayout.ofSignatures(frozenRows.associate { (code, _, hash) -> bySignature.getValue(hash) to code })
        val worse = TailLayoutTestSupport.expectedStates.keys.mapNotNull { name ->
            val n = states(profiles.getValue(name), planned)
            val reference = states(profiles.getValue(name), frozen)
            if (n <= reference + 50) null else "$name: planned=$n frozen=$reference"
        }
        println("planner: ${order.size} signatures, ${(System.nanoTime() - t0) / 1_000_000} ms, same order as table: " +
            (order.map { FieldSignature.hash(it) } == frozenRows.map { it.third }))
        assertEquals(emptyList<String>(), worse)
    }
}
