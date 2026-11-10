package org.nobirds.murmuration.schema

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.serializer
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.tooling.NotBlank
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class KSerializerToJsonSchemaTest {

    @Serializable
    data class Simple(
        @SchemaInfo("user name") val name: String,
        @SchemaInfo("age") val age: Int,
    )

    @Test
    fun `simple data class becomes ObjectSchema with required fields`() {
        val schema = serializer<Simple>().toJsonSchema()
        val obj = schema as ObjectSchema
        assertEquals(setOf("name", "age"), obj.properties.keys)
        assertEquals(listOf("name", "age"), obj.required)

        val nameProp = obj.properties["name"] as StringSchema
        assertEquals("user name", nameProp.description)

        val ageProp = obj.properties["age"] as IntegerSchema
        assertEquals(JsonSchemaFormats.INT32, ageProp.format)
        assertEquals("age", ageProp.description)
    }

    @Serializable
    data class MandatoryWithDefault(
        @SchemaInfo("key") @property:NotBlank val textId: String = "",
        @SchemaInfo("param use-site — invisible to the descriptor") @NotBlank val ignored: String = "",
        @SchemaInfo("optional") val note: String = "",
    )

    @Test
    fun `@property NotBlank makes a field with a default required, param-use-site is a no-op`() {
        val obj = serializer<MandatoryWithDefault>().toJsonSchema() as ObjectSchema

        // A default in Kotlin is a matter of DESERIALIZATION, not of the contract with the model: @NotBlank
        // ("required, must not be empty") must go into required, otherwise constrained
        // decoding legitimately omits the field (incident addAction.textId 2026-07-31).
        assertEquals(listOf("textId"), obj.required)
        val key = obj.properties["textId"] as StringSchema
        assertEquals(1L, key.minLength)
        assertTrue(key.description!!.contains("required, must not be empty"))

        // A bare annotation on a constructor property lands on the PARAMETER (VALUE_PARAMETER in
        // the targets) and does not reach the descriptor — neither required nor minLength. The test pins
        // the trap: on DTOs we write only `@property:`.
        assertTrue("ignored" !in obj.required)
        assertNull((obj.properties["ignored"] as StringSchema).minLength)
    }

    @Test
    fun `unannotated property is excluded from properties and required`() {
        @Serializable
        data class WithUnannotated(
            @SchemaInfo("annotated") val a: String,
            val b: String,
        )

        val obj = serializer<WithUnannotated>().toJsonSchema() as ObjectSchema
        assertEquals(setOf("a"), obj.properties.keys)
        assertEquals(listOf("a"), obj.required)
    }

    // ─── empty/Map guards (strict-only) ─────────────────────────────────────────────────
    // lenient removed: the schema is built strictly both for output and for display. A class whose all
    // fields are filtered out, or a Map field, must fail at build time (this is the signal "the type goes out
    // with an empty/unfit schema"). Display types are caught before prod by a scan test over JsonSchemable.

    @Serializable
    data class AllUnannotated(val a: String, val b: Int)

    @Test
    fun `an empty object schema trips the fail-fast guard`() {
        val err = kotlin.runCatching { serializer<AllUnannotated>().toJsonSchema() }.exceptionOrNull()
        assertNotNull(err)
        assertTrue(err.message!!.contains("EMPTY object schema"))
    }

    @Serializable
    data class WithMap(@SchemaInfo("m") val m: Map<String, String>)

    @Test
    fun `a Map field fails — not expressible in a strict schema`() {
        assertNotNull(kotlin.runCatching { serializer<WithMap>().toJsonSchema() }.exceptionOrNull())
    }

    @Test
    fun `optional property has default and is excluded from required`() {
        @Serializable
        data class WithOptional(
            @SchemaInfo("required field") val a: String,
            @SchemaInfo("optional field") val b: String = "default",
        )

        val obj = serializer<WithOptional>().toJsonSchema() as ObjectSchema
        assertEquals(setOf("a", "b"), obj.properties.keys)
        assertEquals(listOf("a"), obj.required)
    }

    @Test
    fun `nullable property is reflected in nullable flag`() {
        @Serializable
        data class Nullable(
            @SchemaInfo("nullable description") val v: String?,
        )

        val obj = serializer<Nullable>().toJsonSchema() as ObjectSchema
        val v = obj.properties["v"] as StringSchema
        assertTrue(v.nullable, "nullable kotlinx-property → nullable=true on JsonSchema")
    }

    @Test
    fun `Long maps to integer with int64 format`() {
        @Serializable data class L(@SchemaInfo("v") val v: Long)
        val obj = serializer<L>().toJsonSchema() as ObjectSchema
        val v = obj.properties["v"] as IntegerSchema
        assertEquals(JsonSchemaFormats.INT64, v.format)
    }

    @Test
    fun `Double and Float map to number with corresponding format`() {
        @Serializable data class N(@SchemaInfo("d") val d: Double, @SchemaInfo("f") val f: Float)
        val obj = serializer<N>().toJsonSchema() as ObjectSchema
        assertEquals(JsonSchemaFormats.DOUBLE, (obj.properties["d"] as NumberSchema).format)
        assertEquals(JsonSchemaFormats.FLOAT, (obj.properties["f"] as NumberSchema).format)
    }

    @Test
    fun `Boolean maps to BooleanSchema`() {
        @Serializable data class B(@SchemaInfo("flag") val v: Boolean)
        val obj = serializer<B>().toJsonSchema() as ObjectSchema
        assertTrue(obj.properties["v"] is BooleanSchema)
    }

    @Serializable
    enum class Color { RED, GREEN, BLUE }

    @Test
    fun `enum maps to StringSchema with enum values`() {
        @Serializable data class C(@SchemaInfo("c") val v: Color)
        val obj = serializer<C>().toJsonSchema() as ObjectSchema
        val v = obj.properties["v"] as StringSchema
        assertEquals(listOf("RED", "GREEN", "BLUE"), v.enumValues)
    }

    @Serializable
    @SchemaInfo("Legend: Ye=YES, No=NO.")
    enum class Answer { @SerialName("Ye") YES, @SerialName("No") NO }

    @Test
    fun `enum class legend is appended to the description of the place, not shadowed by it`() {
        @Serializable data class A(@SchemaInfo("Own words.") val own: Answer)
        val obj = serializer<A>().toJsonSchema() as ObjectSchema
        val own = obj.properties["own"] as StringSchema
        assertEquals(listOf("Ye", "No"), own.enumValues)
        assertEquals("Own words. Legend: Ye=YES, No=NO.", own.description)
    }

    @Test
    fun `List of primitives maps to ArraySchema with items`() {
        @Serializable data class L(@SchemaInfo("items") val items: List<String>)
        val obj = serializer<L>().toJsonSchema() as ObjectSchema
        val items = obj.properties["items"] as ArraySchema
        assertTrue(items.items is StringSchema)
    }

    @Test
    fun `nested data class becomes nested ObjectSchema`() {
        @Serializable data class Inner(@SchemaInfo("x") val x: Int)
        @Serializable data class Outer(@SchemaInfo("inner") val inner: Inner)
        val obj = serializer<Outer>().toJsonSchema() as ObjectSchema
        val inner = obj.properties["inner"] as ObjectSchema
        assertEquals(setOf("x"), inner.properties.keys)
    }

    @Test
    fun `SchemaInfo metadata propagates to property schema`() {
        @Serializable data class Meta(
            @SchemaInfo(description = "v", example = "abc", format = "uuid", pattern = "[a-z]+", minLength = 1, maxLength = 100)
            val v: String,
        )
        val obj = serializer<Meta>().toJsonSchema() as ObjectSchema
        val v = obj.properties["v"] as StringSchema
        assertEquals("v", v.description)
        assertEquals("abc", v.example)
        assertEquals("uuid", v.format)
        assertEquals("[a-z]+", v.pattern)
        assertEquals(1L, v.minLength)
        assertEquals(100L, v.maxLength)
    }

    @Test
    fun `IntegerSchema picks up min and max from SchemaInfo`() {
        @Serializable data class Bounded(
            @SchemaInfo("v", minValue = 0.0, maxValue = 100.0) val v: Int,
        )
        val obj = serializer<Bounded>().toJsonSchema() as ObjectSchema
        val v = obj.properties["v"] as IntegerSchema
        assertEquals(0L, v.minValue)
        assertEquals(100L, v.maxValue)
    }

    @Serializable
    sealed interface Shape {
        @Serializable
        data class Circle(@SchemaInfo("r") val r: Double) : Shape

        @Serializable
        data class Square(@SchemaInfo("side") val side: Double) : Shape
    }

    @Test
    fun `sealed becomes AnyOfSchema with discriminator-bearing variants`() {
        val schema = serializer<Shape>().toJsonSchema() as AnyOfSchema
        assertEquals(2, schema.variants.size)
        // Each variant is an ObjectSchema with a "type" discriminator property carrying the
        // serial name as enum.
        val firstVariant = schema.variants.first() as ObjectSchema
        val typeProp = firstVariant.properties["type"] as StringSchema
        assertNotNull(typeProp.enumValues, "discriminator property must have an enum")
        assertEquals(1, typeProp.enumValues!!.size)
    }

    @Test
    fun `SchemaGeneratorMeta routes to custom generator`() {
        @Serializable data class Holder(
            @property:SchemaGeneratorMeta(generator = PrimitiveUnionGenerator::class)
            @SchemaInfo("any-primitive")
            val v: String,
        )
        val obj = serializer<Holder>().toJsonSchema() as ObjectSchema
        val v = obj.properties["v"] as AnyOfSchema
        assertEquals(4, v.variants.size, "PrimitiveUnion has 4 variants: string/int/number/bool")
        assertNull(v.description, "AnyOfSchema with no description is fine even when SchemaInfo had one — Vertex disallows it on top of anyOf")
    }
}
