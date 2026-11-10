package org.nobirds.murmuration.llm

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.ThinkingLevel

/**
 * [Llm.clampThinkingLevel] semantics: "closest supported tier not above the requested one,
 * else the lowest supported tier". Regression coverage for the `gemini-3.1-flash-lite-image`
 * catalog fact (live probes 2026-07-15): LOW is rejected with a 400, only MINIMAL and HIGH are
 * accepted — declared as `thinking { level(MINIMAL, HIGH) }` in [GeminiLlms].
 */
class LlmCapabilitiesTest {

    private val liteImageLikeLlm = llm("test", "lite-image-like") {
        thinking { level(ThinkingLevel.MINIMAL, ThinkingLevel.HIGH) }
    }

    private val noConstraintLlm = llm("test", "no-constraint") {
        thinking { level() }
    }

    private val noThinkingLlm = llm("test", "no-thinking") { }

    @Test
    fun `LOW clamps down to MINIMAL when only MINIMAL and HIGH are supported`() {
        assertEquals(ThinkingLevel.MINIMAL, liteImageLikeLlm.clampThinkingLevel(ThinkingLevel.LOW))
    }

    @Test
    fun `MEDIUM clamps down to MINIMAL when only MINIMAL and HIGH are supported`() {
        // MEDIUM (ordinal 2) sits between MINIMAL (0) and HIGH (3) — the nearest tier NOT
        // above it is MINIMAL, not HIGH (HIGH is above, not below).
        assertEquals(ThinkingLevel.MINIMAL, liteImageLikeLlm.clampThinkingLevel(ThinkingLevel.MEDIUM))
    }

    @Test
    fun `HIGH passes through unchanged when explicitly supported`() {
        assertEquals(ThinkingLevel.HIGH, liteImageLikeLlm.clampThinkingLevel(ThinkingLevel.HIGH))
    }

    @Test
    fun `MINIMAL passes through unchanged when explicitly supported`() {
        assertEquals(ThinkingLevel.MINIMAL, liteImageLikeLlm.clampThinkingLevel(ThinkingLevel.MINIMAL))
    }

    @Test
    fun `requested tier below every supported tier falls back to the lowest supported one`() {
        val higherFloor = llm("test", "higher-floor") {
            thinking { level(ThinkingLevel.MEDIUM, ThinkingLevel.HIGH) }
        }
        // Nothing supported is below MINIMAL, so the fallback is the lowest supported tier.
        assertEquals(ThinkingLevel.MEDIUM, higherFloor.clampThinkingLevel(ThinkingLevel.MINIMAL))
    }

    @Test
    fun `no declared tier constraint passes the level through unchanged`() {
        assertEquals(ThinkingLevel.LOW, noConstraintLlm.clampThinkingLevel(ThinkingLevel.LOW))
    }

    @Test
    fun `model without any Thinking capability passes the level through unchanged`() {
        assertEquals(ThinkingLevel.LOW, noThinkingLlm.clampThinkingLevel(ThinkingLevel.LOW))
    }
}
