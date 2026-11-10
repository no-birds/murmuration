package org.nobirds.murmuration.tooling

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNames
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import kotlinx.serialization.json.addJsonObject
import kotlinx.serialization.json.add
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.InstrumentDeclaration
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * [ToolArgumentDecoder] is the one path from wire arguments to typed values: the same enum rules
 * (case-insensitive, `@JsonNames`, no silent default) for a direct, nested, list, map, sealed and
 * nullable enum place; every miss of a call in one [ToolArgumentException]; a case-ambiguous enum
 * refused at mount.
 *
 * Canonical pattern: mount a probe through [InstrumentFactory], feed `prepareWireArguments` a wire
 * object, assert the typed result or the refusal text.
 */
@OptIn(ExperimentalSerializationApi::class)
class ToolArgumentDecoderTest {

    @Serializable
    enum class Dice {
        @SerialName("D4") FOUR,
        @SerialName("D6") @JsonNames("six") SIX,
        @SerialName("D20") TWENTY,
    }

    @Serializable
    data class Roll(val diceType: Dice = Dice.SIX, val count: Int = 1)

    @Serializable
    sealed interface Effect {
        @Serializable @SerialName("damage") data class Damage(val dice: Dice) : Effect
        @Serializable @SerialName("heal") data class Heal(val amount: Int) : Effect
    }

    @Suppress("unused")
    class Probe {
        var last: List<Any?> = emptyList()

        @Instrument("Everything an enum can sit in")
        fun roll(
            @InstrumentParam("direct") dice: Dice? = null,
            @InstrumentParam("nested list") rolls: List<Roll>? = null,
            @InstrumentParam("list") dices: List<Dice>? = null,
            @InstrumentParam("map values") byName: Map<String, Dice>? = null,
            @InstrumentParam("map keys") byDice: Map<Dice, Int>? = null,
            @InstrumentParam("sealed") effect: Effect? = null,
            @InstrumentParam("integral") times: Int = 1,
        ): String {
            last = listOf(dice, rolls, dices, byName, byDice, effect, times)
            return "ok"
        }
    }

    @Serializable
    enum class Color { @SerialName("R") RED, @SerialName("G") GREEN }

    @Serializable
    data class Box<T>(val value: T)

    @Serializable
    data class TwoBoxes(val dice: Box<Dice>, val color: Box<Color>)

    @Suppress("unused")
    class GenericProbe {
        var last: TwoBoxes? = null

        @Instrument("Generic wrapper over two different enums")
        fun pick(@InstrumentParam("boxes") boxes: TwoBoxes): String {
            last = boxes
            return "ok"
        }
    }

    enum class Ambiguous { Fire, FIRE }

    @Serializable
    data class HoldsAmbiguous(val element: Ambiguous)

    @Suppress("unused")
    class AmbiguousProbe {
        @Instrument("Nested ambiguous enum")
        fun cast(@InstrumentParam("spell") spell: HoldsAmbiguous): String = "ok"
    }

    private val probe = Probe()
    private val tool: InstrumentDeclaration<*> = InstrumentFactory().create(probe).instruments.single { it.name == "roll" }

    private fun call(wire: Map<String, JsonElement>): List<Any?> = runBlocking {
        tool.invoke(tool.prepareWireArguments(wire))
        probe.last
    }

    @Test
    fun `direct enum is matched case-insensitively and through JsonNames`() {
        assertEquals(Dice.TWENTY, call(mapOf("dice" to JsonPrimitive("d20")))[0])
        assertEquals(Dice.SIX, call(mapOf("dice" to JsonPrimitive("SIX")))[0])
        assertNull(call(mapOf("dice" to JsonNull))[0], "nullable enum keeps null")
    }

    @Test
    fun `nested, list, map and sealed enum places follow the same rules`() {
        val result = call(buildJsonObject {
            putJsonArray("rolls") { addJsonObject { put("diceType", "d4") } }
            putJsonArray("dices") { add("Six"); add("d20") }
            putJsonObject("byName") { put("a", "d6") }
            putJsonObject("byDice") { put("d20", 2) }
            putJsonObject("effect") { put("type", "damage"); put("dice", "d4") }
        })
        assertEquals(listOf(Roll(Dice.FOUR)), result[1])
        assertEquals(listOf(Dice.SIX, Dice.TWENTY), result[2])
        assertEquals(mapOf("a" to Dice.SIX), result[3])
        assertEquals(mapOf(Dice.TWENTY to 2), result[4])
        assertEquals(Effect.Damage(Dice.FOUR), result[5])
    }

    @Test
    fun `unknown nested enum is refused, not replaced by the field default`() {
        val error = assertFailsWith<ToolArgumentException> {
            tool.prepareWireArguments(buildJsonObject { putJsonArray("rolls") { addJsonObject { put("diceType", "D21") } } })
        }
        assertEquals(
            "parameter 'rolls[0].diceType': unknown value \"D21\". Allowed values: D4, D6, D20.",
            error.message,
        )
    }

    @Test
    fun `absent field still gets its default`() {
        assertEquals(listOf(Roll(Dice.SIX, 3)), call(buildJsonObject {
            putJsonArray("rolls") { addJsonObject { put("count", 3) } }
        })[1])
    }

    @Test
    fun `every miss of the call lands in one refusal`() {
        val error = assertFailsWith<ToolArgumentException> {
            tool.prepareWireArguments(buildJsonObject {
                put("dice", "d3")
                putJsonArray("dices") { add("d6"); add("d8") }
                putJsonObject("byDice") { put("d12", 1) }
                putJsonObject("effect") { put("type", "curse") }
            })
        }
        assertEquals(4, error.problems.size, error.message)
        assertTrue(error.problems.any { it.startsWith("parameter 'dice': unknown value \"d3\"") })
        assertTrue(error.problems.any { it.startsWith("parameter 'dices[1]': unknown value \"d8\"") })
        assertTrue(error.problems.any { it.startsWith("parameter 'byDice (key)': unknown value \"d12\"") })
        assertTrue(error.problems.any { it == "parameter 'effect.type': unknown value \"curse\". Allowed values: damage, heal." })
    }

    @Test
    fun `wrong shape is a ToolArgumentException naming the parameter`() {
        val error = assertFailsWith<ToolArgumentException> {
            tool.prepareWireArguments(buildJsonObject { put("times", 2.5) })
        }
        assertTrue(error.message!!.startsWith("parameter 'times':"), error.message)
    }

    @Test
    fun `whole number sent as a double is accepted for an integral parameter`() {
        assertEquals(5, call(mapOf("times" to JsonPrimitive(5.0)))[6])
    }

    @Test
    fun `case-ambiguous enum reachable from a parameter fails mounting`() {
        val error = assertFailsWith<IllegalArgumentException> { InstrumentFactory().create(AmbiguousProbe()) }
        assertTrue(error.message!!.contains("case-ambiguous"), error.message)
    }

    @Test
    fun `undeclared keys pass through untouched`() {
        val prepared = tool.prepareWireArguments(buildJsonObject { put("extra", "x"); put("dice", "d4") }.jsonObject)
        assertEquals("x", prepared["extra"])
        assertEquals(Dice.FOUR, prepared["dice"])
    }

    @Test
    fun `generic instantiations with different enums each keep their allowed values`() {
        val generic = GenericProbe()
        val pick = InstrumentFactory().create(generic).instruments.single { it.name == "pick" }
        fun wire(dice: String, color: String) = buildJsonObject {
            putJsonObject("boxes") {
                putJsonObject("dice") { put("value", dice) }
                putJsonObject("color") { put("value", color) }
            }
        }.jsonObject

        runBlocking { pick.invoke(pick.prepareWireArguments(wire("d4", "g"))) }
        assertEquals(TwoBoxes(Box(Dice.FOUR), Box(Color.GREEN)), generic.last)

        val error = assertFailsWith<ToolArgumentException> { pick.prepareWireArguments(wire("d4", "blue")) }
        assertTrue(error.problems.single().contains("Allowed values: R, G"), error.message)
    }
}
