package org.nobirds.murmuration.gemini.fst

import com.google.genai.types.FunctionDeclaration
import com.google.genai.types.Schema
import com.google.genai.types.Type
import org.nobirds.murmuration.gemini.fst.FstSymbols.DIGIT
import org.nobirds.murmuration.gemini.fst.FstSymbols.ESCAPE_FOLLOW
import org.nobirds.murmuration.gemini.fst.FstSymbols.HEX
import org.nobirds.murmuration.gemini.fst.FstSymbols.STRING_CHAR
import java.util.BitSet

/**
 * Builds the NFA of the language of all valid function calls `name{key:value,...}` for a set of
 * declarations. The call syntax is the one fitted to measured Vertex points: unquoted keys, a
 * string body that accepts `\X` and `\uXX`, optional leading `-` on integers, `e`/`E` exponents
 * on numbers. Port of the offline Python reference model (`fstmodel.py`); the two must stay in step.
 */
internal class FstLanguageBuilder {
    private val nfa = Nfa()

    /** The NFA accepting any one call of [declarations], and its start and final states. */
    fun build(declarations: List<FunctionDeclaration>): Built {
        val start = nfa.newState()
        val final = nfa.newState()
        for (declaration in declarations) {
            val name = declaration.name().orElseThrow { IllegalArgumentException("Function declaration without a name") }
            val afterName = nfa.literal(start, name)
            val parameters = declaration.parameters().orElse(null)
            val end = if (parameters != null && parameters.properties().orElse(emptyMap()).isNotEmpty()) {
                buildObject(afterName, parameters, "$name.parameters")
            } else {
                nfa.literal(afterName, "{}")
            }
            nfa.addEpsilon(end, final)
        }
        return Built(nfa, BitSet().also { it.set(start) }, BitSet().also { it.set(final) })
    }

    class Built(val nfa: Nfa, val starts: BitSet, val finals: BitSet)

    private fun buildValue(from: Int, schema: Schema, place: String): Int {
        val end = nfa.newState()
        fun alt(builder: (Int) -> Int) = nfa.addEpsilon(builder(from), end)

        if (schema.nullable().orElse(false)) alt { nfa.literal(it, "null") }
        val variants = schema.anyOf().orElse(null)
        if (variants != null) {
            variants.forEachIndexed { i, v ->
                if (v.type().orElse(null)?.knownEnum() == Type.Known.NULL) {
                    alt { nfa.literal(it, "null") }
                } else {
                    alt { buildValue(it, v, "$place.anyOf[$i]") }
                }
            }
            return end
        }
        when (val type = schema.type().orElse(null)?.knownEnum() ?: Type.Known.STRING) {
            Type.Known.STRING -> {
                val values = schema.enum_().orElse(emptyList())
                if (values.isNotEmpty()) {
                    for (value in values) alt { nfa.literal(it, "\"$value\"") }
                } else {
                    alt(::buildString)
                }
            }
            Type.Known.INTEGER -> alt { buildInteger(it, schema.minimum().orElse(null)) }
            Type.Known.NUMBER -> alt(::buildNumber)
            Type.Known.BOOLEAN -> {
                alt { nfa.literal(it, "true") }
                alt { nfa.literal(it, "false") }
            }
            Type.Known.ARRAY -> alt { buildArray(it, schema, place) }
            Type.Known.OBJECT -> alt { buildObject(it, schema, place) }
            else -> throw IllegalArgumentException("FST model does not know type $type at $place")
        }
        return end
    }

    private fun buildString(from: Int): Int {
        val loop = nfa.literal(from, "\"")
        nfa.add(loop, STRING_CHAR, loop)
        val escape = nfa.newState()
        nfa.add(loop, '\\'.code, escape)
        nfa.add(escape, ESCAPE_FOLLOW, loop)
        val hex1 = nfa.literal(escape, "u")
        val hex2 = nfa.newState()
        nfa.add(hex1, HEX, hex2)
        nfa.add(hex2, HEX, loop)
        return nfa.literal(loop, "\"")
    }

    private fun buildInteger(from: Int, minimum: Double?): Int {
        var start = from
        if (!(minimum != null && minimum >= 0)) {
            val signed = nfa.newState()
            nfa.add(from, '-'.code, signed)
            nfa.addEpsilon(from, signed)
            start = signed
        }
        val out = nfa.newState()
        val zero = nfa.newState()
        nfa.add(start, '0'.code, zero)
        if (!(minimum != null && minimum >= 1)) nfa.addEpsilon(zero, out)
        val digits = nfa.newState()
        nfa.add(start, DIGIT, digits)
        nfa.add(digits, DIGIT, digits)
        nfa.add(digits, '0'.code, digits)
        nfa.addEpsilon(digits, out)
        return out
    }

    private fun buildNumber(from: Int): Int {
        val signed = nfa.newState()
        nfa.add(from, '-'.code, signed)
        nfa.addEpsilon(from, signed)
        var out = nfa.newState()
        var digits = nfa.newState()
        val zero = nfa.newState()
        nfa.add(signed, '0'.code, zero)
        nfa.add(signed, DIGIT, digits)
        nfa.add(digits, DIGIT, digits)
        nfa.add(digits, '0'.code, digits)
        val integerPart = nfa.newState()
        nfa.addEpsilon(zero, integerPart)
        nfa.addEpsilon(digits, integerPart)
        digits = integerPart
        nfa.addEpsilon(digits, out)
        // fraction
        val point = nfa.newState()
        nfa.add(digits, '.'.code, point)
        val fraction = nfa.newState()
        nfa.add(point, DIGIT, fraction)
        nfa.add(point, '0'.code, fraction)
        nfa.add(fraction, DIGIT, fraction)
        nfa.add(fraction, '0'.code, fraction)
        nfa.addEpsilon(fraction, out)
        // exponent
        val exp = nfa.newState()
        nfa.add(out, 'e'.code, exp)
        nfa.add(out, 'E'.code, exp)
        val sign = nfa.newState()
        nfa.addEpsilon(exp, sign)
        nfa.add(exp, '+'.code, sign)
        nfa.add(exp, '-'.code, sign)
        val expDigits = nfa.newState()
        nfa.add(sign, DIGIT, expDigits)
        nfa.add(sign, '0'.code, expDigits)
        nfa.add(expDigits, DIGIT, expDigits)
        nfa.add(expDigits, '0'.code, expDigits)
        val afterExp = nfa.newState()
        nfa.addEpsilon(out, afterExp)
        nfa.addEpsilon(expDigits, afterExp)
        out = afterExp
        return out
    }

    private fun buildArray(from: Int, schema: Schema, place: String): Int {
        val items = schema.items().orElseGet { Schema.builder().build() }
        val open = nfa.literal(from, "[")
        val close = nfa.newState()
        if ((schema.minItems().orElse(0L)) < 1) nfa.addEpsilon(open, close)
        val firstItemEnd = buildValue(open, items, "$place[]")
        nfa.addEpsilon(firstItemEnd, close)
        val comma = nfa.literal(firstItemEnd, ",")
        val nextItemEnd = buildValue(comma, items, "$place[]")
        nfa.addEpsilon(nextItemEnd, firstItemEnd)
        nfa.addEpsilon(nextItemEnd, close)
        return nfa.literal(close, "]")
    }

    private fun buildObject(from: Int, schema: Schema, place: String): Int {
        val properties = schema.properties().orElse(emptyMap())
        val required = schema.required().orElse(emptyList()).toSet()
        val afterOpen = nfa.literal(from, "{")
        // `first`: nothing emitted yet; `rest`: at least one key emitted. Null = unreachable.
        var first: Int? = afterOpen
        var rest: Int? = null
        for (key in properties.keys.sortedWith(CODE_POINT_ORDER)) {
            val keyText = "$key:"
            val nextRest = nfa.newState()
            var nextFirst: Int? = null
            val value = properties.getValue(key)
            first?.let { nfa.addEpsilon(buildValue(nfa.literal(it, keyText), value, "$place.$key"), nextRest) }
            rest?.let { nfa.addEpsilon(buildValue(nfa.literal(it, ",$keyText"), value, "$place.$key"), nextRest) }
            if (key !in required) {
                nextFirst = first
                rest?.let { nfa.addEpsilon(it, nextRest) }
            }
            first = nextFirst
            rest = nextRest
        }
        val close = nfa.newState()
        first?.let { nfa.addEpsilon(it, close) }
        rest?.let { nfa.addEpsilon(it, close) }
        return nfa.literal(close, "}")
    }

    private companion object {
        val CODE_POINT_ORDER = Comparator<String> { a, b ->
            val x = a.codePoints().toArray()
            val y = b.codePoints().toArray()
            val n = minOf(x.size, y.size)
            for (i in 0 until n) if (x[i] != y[i]) return@Comparator x[i].compareTo(y[i])
            x.size.compareTo(y.size)
        }
    }
}
