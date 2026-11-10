@file:OptIn(ExperimentalSerializationApi::class)

package org.nobirds.murmuration.tooling

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.KSerializer
import kotlinx.serialization.descriptors.PolymorphicKind
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.SerialKind
import kotlinx.serialization.descriptors.StructureKind
import kotlinx.serialization.encoding.CompositeDecoder
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNames
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import org.nobirds.murmuration.schema.parameterSerializer
import org.nobirds.murmuration.schema.sealedClassDiscriminator
import kotlin.reflect.KType

/**
 * THE single path from a provider's raw tool-call arguments to typed Kotlin values, for every
 * parameter of a reflective tool — direct or nested, enum or not, on every provider.
 *
 * For each declared parameter it
 * 1. walks the [JsonElement] along the parameter's [SerialDescriptor] (class properties, list
 *    items, map keys and values, the sealed subclass chosen by its discriminator — the same rule
 *    the schema uses, `sealedClassDiscriminator`; nullable descriptors are walked as their base);
 * 2. matches every enum place through [AllowedValues] — case-insensitively, `@JsonNames` included —
 *    and rewrites it to the canonical code;
 * 3. collects every miss of the call;
 * 4. only then runs `decodeFromJsonElement`.
 *
 * Any miss or decode failure becomes ONE [ToolArgumentException] listing all problems of the call,
 * each with the allowed values where it is an enum. An unknown enum value is therefore never
 * replaced by a default: `coerceInputValues` stays on only for `null → default` of non-enum fields,
 * the enum places reach kotlinx already canonical.
 *
 * This is NOT a [ParameterValidator]: a validator's `Reject` is a successful string result of the
 * call, produced after decoding; an argument that cannot be decoded at all has no typed value to
 * validate, so it is refused by an exception before the tool runs. The two channels are separate on
 * purpose.
 *
 * Every enum reachable from the parameters gets its [AllowedValues] at construction, i.e. at tool
 * mount, so a case-ambiguous enum fails mounting, not a call.
 */
class ToolArgumentDecoder(parameters: List<Pair<String, KType>>) {

    private val serializers: Map<String, KSerializer<Any?>> =
        parameters.associate { (name, type) -> name to type.parameterSerializer() }

    private val allowedValues: Map<String, AllowedValues> = buildMap {
        val visited = mutableSetOf<SerialDescriptor>()
        serializers.values.forEach { collectEnums(it.descriptor, this, visited) }
    }

    /** Names of the parameters this decoder owns. */
    val parameterNames: Set<String> get() = serializers.keys

    /**
     * Decodes the arguments of the declared parameters present in [args]; keys of other parameters
     * are not this decoder's business and are ignored. A JSON `null` stays `null`.
     *
     * @throws ToolArgumentException with every problem of the call.
     */
    fun decode(args: Map<String, JsonElement>): Map<String, Any?> {
        val problems = mutableListOf<String>()
        val normalized = args.filterKeys { it in serializers }
            .mapValues { (name, value) -> normalize(value, serializers.getValue(name).descriptor, name, problems) }
        if (problems.isNotEmpty()) throw ToolArgumentException(problems)

        val decoded = normalized.mapValues { (name, value) ->
            if (value is JsonNull) return@mapValues null
            try {
                json.decodeFromJsonElement(serializers.getValue(name), value)
            } catch (e: IllegalArgumentException) {
                problems += "parameter '$name': ${e.message ?: e::class.simpleName}"
                null
            }
        }
        if (problems.isNotEmpty()) throw ToolArgumentException(problems)
        return decoded
    }

    private fun normalize(element: JsonElement, descriptor: SerialDescriptor, path: String, problems: MutableList<String>): JsonElement {
        if (element is JsonNull) return element
        if (descriptor.isInline && descriptor.elementsCount == 1) {
            return normalize(element, descriptor.getElementDescriptor(0), path, problems)
        }
        return when (descriptor.kind) {
            SerialKind.ENUM -> normalizeEnum(element, descriptor, path, problems)
            StructureKind.CLASS, StructureKind.OBJECT ->
                (element as? JsonObject)?.let { normalizeProperties(it, descriptor, path, problems) } ?: element
            StructureKind.LIST -> (element as? JsonArray)?.let { array ->
                val itemDescriptor = descriptor.getElementDescriptor(0)
                JsonArray(array.mapIndexed { index, item -> normalize(item, itemDescriptor, "$path[$index]", problems) })
            } ?: element
            StructureKind.MAP -> (element as? JsonObject)?.let { map ->
                val keyDescriptor = descriptor.getElementDescriptor(0)
                val valueDescriptor = descriptor.getElementDescriptor(1)
                JsonObject(map.entries.associate { (key, value) ->
                    val normalizedKey = if (keyDescriptor.kind == SerialKind.ENUM) {
                        (normalizeEnum(JsonPrimitive(key), keyDescriptor, "$path (key)", problems) as JsonPrimitive).content
                    } else {
                        key
                    }
                    normalizedKey to normalize(value, valueDescriptor, "$path[$key]", problems)
                })
            } ?: element
            PolymorphicKind.SEALED -> (element as? JsonObject)?.let { normalizeSealed(it, descriptor, path, problems) } ?: element
            PrimitiveKind.BYTE, PrimitiveKind.SHORT, PrimitiveKind.INT, PrimitiveKind.LONG -> element.integralOrSelf()
            else -> element
        }
    }

    private fun normalizeEnum(element: JsonElement, descriptor: SerialDescriptor, path: String, problems: MutableList<String>): JsonElement {
        val primitive = element as? JsonPrimitive ?: return element
        val allowed = allowedValues.getValue(descriptor.enumKey())
        val canonical = allowed.canonicalOf(primitive.content)
        if (canonical == null) {
            problems += "parameter '$path': unknown value \"${primitive.content}\". Allowed values: ${allowed.render()}."
            return element
        }
        return JsonPrimitive(canonical)
    }

    private fun normalizeProperties(obj: JsonObject, descriptor: SerialDescriptor, path: String, problems: MutableList<String>): JsonObject =
        JsonObject(obj.mapValues { (key, value) ->
            val index = descriptor.propertyIndexOf(key) ?: return@mapValues value
            normalize(value, descriptor.getElementDescriptor(index), "$path.$key", problems)
        })

    private fun normalizeSealed(obj: JsonObject, descriptor: SerialDescriptor, path: String, problems: MutableList<String>): JsonElement {
        val discriminator = descriptor.sealedClassDiscriminator(descriptor.getElementAnnotations(1))
        val tag = (obj[discriminator] as? JsonPrimitive)?.content ?: return obj
        val subclasses = descriptor.getElementDescriptor(1)
        val tags = (0 until subclasses.elementsCount).map { subclasses.getElementName(it) }
        val index = tags.indexOf(tag)
        if (index < 0) {
            problems += "parameter '$path.$discriminator': unknown value \"$tag\". Allowed values: ${tags.joinToString(", ")}."
            return obj
        }
        return normalizeProperties(obj, subclasses.getElementDescriptor(index), path, problems)
    }

    private fun collectEnums(descriptor: SerialDescriptor, into: MutableMap<String, AllowedValues>, visited: MutableSet<SerialDescriptor>) {
        if (descriptor.kind == SerialKind.ENUM) {
            into.getOrPut(descriptor.enumKey()) { AllowedValues.of(descriptor) }
            return
        }
        // Only named types can recurse; collections share one serial name (`ArrayList`) whatever
        // their item type, so they are always walked.
        val named = descriptor.kind == StructureKind.CLASS || descriptor.kind == StructureKind.OBJECT ||
            descriptor.kind == PolymorphicKind.SEALED
        // Keyed by the descriptor, not its serial name: `Box<A>` and `Box<B>` share one serial name
        // but differ in type-parameter descriptors, so each instantiation is walked.
        if (named && !visited.add(descriptor)) return
        for (index in 0 until descriptor.elementsCount) {
            collectEnums(descriptor.getElementDescriptor(index), into, visited)
        }
    }

    private companion object {
        /** The decoding configuration tool arguments always had (the former `Instrument.defaultJson`). */
        val json = Json {
            ignoreUnknownKeys = true
            isLenient = true
            coerceInputValues = true
        }

        fun SerialDescriptor.enumKey(): String = serialName.removeSuffix("?")

        /** Property index by its serial name or a [JsonNames] alias, as kotlinx resolves keys. */
        fun SerialDescriptor.propertyIndexOf(key: String): Int? {
            val direct = getElementIndex(key)
            if (direct != CompositeDecoder.UNKNOWN_NAME) return direct
            return (0 until elementsCount).firstOrNull { index ->
                getElementAnnotations(index).filterIsInstance<JsonNames>().any { key in it.names }
            }
        }

        /**
         * A whole number sent in floating form (`5.0`) is rewritten to `5` for an integral
         * parameter: some providers render every number as a double, and the former direct-param
         * path accepted it. A fractional value is left as is and refused by kotlinx.
         */
        fun JsonElement.integralOrSelf(): JsonElement {
            val primitive = this as? JsonPrimitive ?: return this
            if (primitive.isString) return this
            val number = primitive.content.toDoubleOrNull() ?: return this
            if (primitive.content.toLongOrNull() != null) return this
            if (number % 1.0 != 0.0 || number !in Long.MIN_VALUE.toDouble()..Long.MAX_VALUE.toDouble()) return this
            return JsonPrimitive(number.toLong())
        }
    }
}
