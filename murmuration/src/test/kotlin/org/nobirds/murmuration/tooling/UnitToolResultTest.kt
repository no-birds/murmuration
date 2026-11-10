package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.prompt.CallIds
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.ToolResultPromptPart
import org.nobirds.murmuration.prompt.toJsonElement
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Regression test for the "kotlin.Unit" tool_result leak. A `Unit`-returning instrument
 * (setters/mutators like updateHp, setSummary, updateToken, ...) used to serialize to the
 * literal string "kotlin.Unit" in the model-facing tool_result. That reads as a garbled or
 * failed result rather than a confirmation — models that make one tool call per round-trip
 * (GPT-5.x) reacted by redoing the already-applied mutation "just in case" (observed as
 * updateHp/setSummary/updateMemory fired 4-5x with identical args within a single turn).
 *
 * The fix lives in [toJsonElement] (org.nobirds.murmuration.prompt.PromptExecution.kt) — the
 * single place where an instrument's raw return value is converted into the [JsonElement]
 * carried by [ToolResultPromptPart.result], which both the OpenAI and Gemini message mappers
 * read as-is (`part.result.toString()` / `part.result.toAny()`).
 */
class UnitToolResultTest {

    class Mutator {
        @Instrument("Applies a mutation and returns nothing meaningful.")
        fun mutate(): Unit = Unit

        @Instrument("Returns a domain string result, must pass through untouched.")
        fun withMessage(): String = "Task 'foo' added."
    }

    /** Mirrors what the OpenAI mapper does with a tool result: `part.result.toString()`. */
    private fun toolResultText(result: Any?): String =
        ToolResultPromptPart("tool", result.toJsonElement(), callId = CallIds.mint()).result.toString()

    @Test
    fun `Unit-returning instrument renders a meaningful ack, not the type name`() = runBlocking {
        val declaration = InstrumentFactory().create(Mutator()).instruments.single { it.name == "mutate" }
        val invocationResult = declaration.invoke(emptyMap())

        val text = toolResultText(invocationResult)

        assertFalse(text.contains("kotlin.Unit"), "tool_result must not leak the Kotlin type name: $text")
        assertTrue(text.isNotBlank(), "tool_result must contain a meaningful acknowledgement")
    }

    @Test
    fun `String-returning instrument passes through unchanged`() = runBlocking {
        val declaration = InstrumentFactory().create(Mutator()).instruments.single { it.name == "withMessage" }
        val invocationResult = declaration.invoke(emptyMap())

        assertEquals("Task 'foo' added.", invocationResult)
        assertTrue(toolResultText(invocationResult).contains("Task 'foo' added."))
    }

    @Test
    fun `null result still renders sensibly, not the Unit leak`() {
        val text = toolResultText(null)
        assertFalse(text.contains("kotlin.Unit"))
    }

    @Test
    fun `Unit toJsonElement is an explicit OK acknowledgement`() {
        assertEquals(JsonPrimitive("OK"), Unit.toJsonElement())
    }
}
