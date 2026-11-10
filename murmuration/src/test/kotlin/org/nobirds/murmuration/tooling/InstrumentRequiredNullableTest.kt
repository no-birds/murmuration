package org.nobirds.murmuration.tooling

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.schema.toFunctionSchema
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Verifies the "required + nullable" idiom we rely on to keep Gemini FST constraint small
 * on partial-update tools: dropping `= null` defaults must promote the parameter from
 * optional (absent from `required[]`) to required + nullable (present in `required[]`,
 * with `"nullable": true` on the property schema).
 */
class InstrumentRequiredNullableTest {

    @Suppress("unused")
    class WithDefaults {
        @Instrument("Update something. All nullable, all defaulted.")
        fun updateThing(
            @InstrumentParam("New name") name: String? = null,
            @InstrumentParam("New description") description: String? = null,
            @InstrumentParam("New count") count: Int? = null,
            @InstrumentParam("New tags") tags: List<String>? = null,
        ): String = "ok"
    }

    @Suppress("unused")
    class WithoutDefaults {
        @Instrument("Update something. All nullable, no defaults.")
        fun updateThing(
            @InstrumentParam("New name") name: String?,
            @InstrumentParam("New description") description: String?,
            @InstrumentParam("New count") count: Int?,
            @InstrumentParam("New tags") tags: List<String>?,
        ): String = "ok"
    }

    @Test
    fun `optional parameters are not in required list and schema has nullable`() {
        val decl = InstrumentFactory().create(WithDefaults()).instruments.single()
        val schema = decl.toFunctionSchema().parameters.toJson()

        val required = schema["required"]?.jsonArray?.map { it.jsonPrimitive.content } ?: emptyList()
        assertEquals(emptyList(), required, "With defaults: required[] must be empty")

        val props = schema["properties"]!!.jsonObject
        listOf("name", "description", "count", "tags").forEach { propName ->
            val prop = props[propName]!!.jsonObject
            assertEquals(true, prop["nullable"]?.jsonPrimitive?.content?.toBoolean(),
                "$propName must have \"nullable\": true")
        }
    }

    @Test
    fun `nullable parameters without defaults appear in required list AND keep nullable`() {
        val decl = InstrumentFactory().create(WithoutDefaults()).instruments.single()
        val schema = decl.toFunctionSchema().parameters.toJson()

        val required = schema["required"]?.jsonArray?.map { it.jsonPrimitive.content } ?: emptyList()
        assertEquals(
            listOf("name", "description", "count", "tags").sorted(),
            required.sorted(),
            "Without defaults: every parameter must be required",
        )

        val props = schema["properties"]!!.jsonObject
        listOf("name", "description", "count", "tags").forEach { propName ->
            val prop = props[propName]!!.jsonObject
            assertEquals(true, prop["nullable"]?.jsonPrimitive?.content?.toBoolean(),
                "$propName must still be \"nullable\": true (required + nullable, not non-null required)")
        }
    }

    @Test
    fun `byte size of without-defaults schema is comparable to with-defaults`() {
        val a = InstrumentFactory().create(WithDefaults()).instruments.single().toFunctionSchema().parameters.toJson().toString()
        val b = InstrumentFactory().create(WithoutDefaults()).instruments.single().toFunctionSchema().parameters.toJson().toString()
        // The without-defaults variant carries a `required[]` array of N names plus quotes/commas.
        // Just sanity-check that it isn't catastrophically larger and isn't smaller.
        assertTrue(b.length >= a.length, "without-defaults schema must be >= with-defaults (we added required[])")
        assertTrue(b.length < a.length + 200, "without-defaults schema must not be drastically larger")
    }

    @Test
    fun `print both schemas for visual inspection`() {
        val a = InstrumentFactory().create(WithDefaults()).instruments.single().toFunctionSchema().parameters.toJson()
        val b = InstrumentFactory().create(WithoutDefaults()).instruments.single().toFunctionSchema().parameters.toJson()
        println("=== WithDefaults ===")
        println(a)
        println("=== WithoutDefaults ===")
        println(b)
    }
}
