@file:OptIn(ExperimentalSerializationApi::class)

package org.nobirds.murmuration.schema

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.KSerializer
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.builtins.SetSerializer
import kotlinx.serialization.builtins.nullable
import kotlinx.serialization.serializer
import org.nobirds.murmuration.internal.UUIDSerializer
import java.util.UUID
import kotlin.reflect.KClass
import kotlin.reflect.KType

/**
 * Resolves the kotlinx serializer for an instrument parameter type. It is
 * `serializer(type)` plus one extra rule: [UUID] is known at ANY nesting level — `UUID`,
 * `UUID?`, `List<UUID>`, `Set<UUID>`, `Map<String, UUID>`, and any combination of those
 * (`List<UUID?>?`, `Map<String, List<UUID>>`). kotlinx has no built-in serializer for
 * `java.util.UUID`, so the plain `serializer(type)` fails with "Serializer not found" as soon
 * as a UUID appears in the type.
 *
 * Types without a UUID anywhere in their type-argument tree are resolved by plain
 * `serializer(type)`, so the behaviour for them is unchanged. A UUID nested in a type argument
 * of a class other than the standard collections is not supported and fails loudly.
 *
 * The same serializer drives both the schema (`{type: string, format: uuid}`) and the decoding
 * of the coerced wire value, so the two cannot disagree on nesting.
 */
fun KType.parameterSerializer(): KSerializer<Any?> {
    @Suppress("UNCHECKED_CAST")
    return resolveSerializer() as KSerializer<Any?>
}

private fun KType.containsUuid(): Boolean =
    classifier == UUID::class || arguments.any { it.type?.containsUuid() == true }

private val LIST_CLASSES: Set<KClass<*>> = setOf(List::class, Collection::class, Iterable::class)
private val SET_CLASSES: Set<KClass<*>> = setOf(Set::class)
private val MAP_CLASSES: Set<KClass<*>> = setOf(Map::class)

private fun KType.resolveSerializer(): KSerializer<*> {
    if (!containsUuid()) return serializer(this)

    val base: KSerializer<*> = when (classifier) {
        UUID::class -> UUIDSerializer
        in LIST_CLASSES -> ListSerializer(typeArgument(0).resolveSerializer())
        in SET_CLASSES -> SetSerializer(typeArgument(0).resolveSerializer())
        in MAP_CLASSES -> MapSerializer(typeArgument(0).resolveSerializer(), typeArgument(1).resolveSerializer())
        else -> throw IllegalArgumentException(
            "UUID nested in a type argument of $this is not supported; " +
                "supported containers are List, Set, Collection, Iterable and Map",
        )
    }
    @Suppress("UNCHECKED_CAST")
    return if (isMarkedNullable) (base as KSerializer<Any>).nullable else base
}

private fun KType.typeArgument(index: Int): KType =
    arguments.getOrNull(index)?.type
        ?: throw IllegalArgumentException("Star-projected type argument #$index in $this has no serializer")
