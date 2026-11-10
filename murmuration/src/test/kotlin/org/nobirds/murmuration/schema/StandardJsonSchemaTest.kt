package org.nobirds.murmuration.schema

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals

class StandardJsonSchemaTest {

    private val schema = ObjectSchema(
        properties = linkedMapOf(
            "a" to StringSchema(nullable = true),
            "b" to AnyOfSchema(listOf(IntegerSchema()), nullable = true),
        ),
        required = listOf("a"),
    )

    @Test
    fun `nullable is a type union, not the OpenAPI flag`() {
        assertEquals(
            """{"type":"object","properties":{"a":{"type":["string","null"]},""" +
                """"b":{"anyOf":[{"type":"integer"},{"type":"null"}]}},"required":["a"]}""",
            schema.toStandardJson().toString(),
        )
    }

    @Test
    fun `closedObjects adds additionalProperties false`() {
        assertEquals(
            """{"type":"object","properties":{"a":{"type":["string","null"]},""" +
                """"b":{"anyOf":[{"type":"integer"},{"type":"null"}]}},"required":["a"],"additionalProperties":false}""",
            schema.toStandardJson(closedObjects = true).toString(),
        )
    }
}
