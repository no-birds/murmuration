package org.nobirds.murmuration.gemini.fst

import com.google.genai.types.FunctionDeclaration
import com.google.genai.types.Schema
import com.google.genai.types.Type
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.int
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Parity of [FstStateModel] with the Python reference on dumped tool profiles (declarations as
 * they go on the wire). The algorithm is the same, so equality is exact, with no tolerance.
 *
 * `expected.json` is N per profile from the reference: for each file in `profiles/`, run
 * `fstmodel.min_states(json.load(file), CFG) + 36` with the Python reference (`python3 -I`).
 */
class FstStateModelParityTest {
    private val expected: Map<String, Int> = resource("fst/expected.json")
        .let { Json.parseToJsonElement(it) as JsonObject }
        .mapValues { it.value.jsonPrimitive.int }

    @Test
    fun `every dumped profile gets exactly the reference N`() {
        assertTrue(expected.size >= 28, "expected.json lists ${expected.size} profiles")
        val mismatches = expected.filter { (name, n) -> states(name) != n }
            .map { (name, n) -> "$name: kotlin=${states(name)} python=$n" }
        assertEquals(emptyList<String>(), mismatches)
    }

    @Test
    fun `empty declaration list is the baseline plus the empty language`() {
        val n = FstStateModel.states(emptyList())
        assertTrue(n >= FstStateModel.BASELINE_STATES, "n=$n")
    }

    @Test
    fun `integer ranges outside the measured ones are rejected`() {
        fun tool(min: Double, max: Double?) = FunctionDeclaration.builder().name("t").parameters(
            Schema.builder().type(Type.Known.OBJECT).properties(
                mapOf("n" to Schema.builder().type(Type.Known.INTEGER).minimum(min).also { b -> max?.let { b.maximum(it) } }.build()),
            ).build(),
        ).build()

        FstStateModel.states(listOf(tool(0.0, 9.0)))
        FstStateModel.states(listOf(tool(1.0, 20.0)))
        val wide = assertFailsWith<IllegalArgumentException> { FstStateModel.states(listOf(tool(0.0, 100.0))) }
        assertTrue("t.parameters.n" in wide.message.orEmpty() && "maximum" in wide.message.orEmpty(), wide.message)
        val negative = assertFailsWith<IllegalArgumentException> { FstStateModel.states(listOf(tool(-1000000.0, null))) }
        assertTrue("minimum" in negative.message.orEmpty(), negative.message)
    }

    @Test
    fun `minItems of two or more is rejected`() {
        fun tool(minItems: Long) = FunctionDeclaration.builder().name("t").parameters(
            Schema.builder().type(Type.Known.OBJECT).properties(
                mapOf("xs" to Schema.builder().type(Type.Known.ARRAY).minItems(minItems)
                    .items(Schema.builder().type(Type.Known.STRING).build()).build()),
            ).build(),
        ).build()

        FstStateModel.states(listOf(tool(0)))
        FstStateModel.states(listOf(tool(1)))
        val error = assertFailsWith<IllegalArgumentException> { FstStateModel.states(listOf(tool(3))) }
        assertTrue("t.parameters.xs" in error.message.orEmpty() && "minItems 3" in error.message.orEmpty(), error.message)
    }

    private fun states(profile: String): Int = FstStateModel.states(declarations(profile))

    private fun declarations(profile: String): List<FunctionDeclaration> =
        (Json.parseToJsonElement(resource("fst/profiles/$profile.json")) as JsonArray)
            .map { FunctionDeclaration.fromJson(it.jsonObject.toString()) }

    private fun resource(path: String): String =
        checkNotNull(javaClass.classLoader.getResourceAsStream(path)) { "no resource $path" }
            .use { it.readBytes().toString(Charsets.UTF_8) }
}
