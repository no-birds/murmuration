@file:OptIn(ExperimentalSerializationApi::class)

package org.nobirds.murmuration.tooling

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.SerialKind
import kotlinx.serialization.json.JsonNames
import java.util.Locale

/**
 * The set of values an enum accepts on the wire, read from its [SerialDescriptor]: the canonical
 * codes ([SerialDescriptor.getElementName], i.e. `@SerialName` or the constant name — exactly what
 * the schema's `enum` lists) plus their [JsonNames] aliases. Matching is case-insensitive and
 * always yields the canonical code.
 *
 * Construction refuses a case-ambiguous enum (two different constants whose codes or aliases differ
 * only by case): with case-insensitive matching such a value would silently go to whichever
 * constant came first. The check runs at tool mount, for every enum reachable from a tool's
 * parameters ([ToolArgumentDecoder]).
 */
class AllowedValues private constructor(
    /** Serial name of the enum, without the nullable `?` suffix. */
    val enumName: String,
    /** Canonical codes in declaration order. */
    val values: List<String>,
    private val canonicalByFolded: Map<String, String>,
) {
    /** The canonical code [raw] stands for (case-insensitive, aliases included), or `null`. */
    fun canonicalOf(raw: String): String? = canonicalByFolded[raw.fold()]

    /** The list shown to the model in a refusal: canonical codes only, as in the schema. */
    fun render(): String = values.joinToString(", ")

    companion object {
        fun of(descriptor: SerialDescriptor): AllowedValues {
            require(descriptor.kind == SerialKind.ENUM) { "${descriptor.serialName} is not an enum descriptor" }
            val enumName = descriptor.serialName.removeSuffix("?")
            val values = (0 until descriptor.elementsCount).map { descriptor.getElementName(it) }
            val canonicalByFolded = mutableMapOf<String, String>()
            values.forEachIndexed { index, canonical ->
                val aliases = descriptor.getElementAnnotations(index).filterIsInstance<JsonNames>().flatMap { it.names.asList() }
                for (name in listOf(canonical) + aliases) {
                    val previous = canonicalByFolded.putIfAbsent(name.fold(), canonical)
                    require(previous == null || previous == canonical) {
                        "Enum '$enumName' is case-ambiguous: \"$name\" of $canonical and a name of $previous " +
                            "differ only by case, so a tool argument could not be matched unambiguously"
                    }
                }
            }
            return AllowedValues(enumName, values, canonicalByFolded)
        }

        private fun String.fold(): String = lowercase(Locale.ROOT)
    }
}
