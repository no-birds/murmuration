package org.nobirds.murmuration.gemini

import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Thinking
import org.nobirds.murmuration.llm.ThinkingMode
import org.nobirds.murmuration.llm.capability
import org.nobirds.murmuration.prompt.ThinkingLevel
import kotlin.test.assertEquals
import kotlin.test.assertNull

class Gemini38ThinkingLevelsTest {

    private fun supported(provider: String, model: String): Set<ThinkingLevel>? =
        GeminiLlms.findModel(provider, model).capability<Thinking>()?.modes
            ?.filterIsInstance<ThinkingMode.Level>()?.firstOrNull()?.supported

    @Test
    fun `3_8 flash declares LOW MEDIUM HIGH on both providers`() {
        val expected = setOf(ThinkingLevel.LOW, ThinkingLevel.MEDIUM, ThinkingLevel.HIGH)
        assertEquals(expected, supported("vertex", "gemini-3.8-flash"))
        assertEquals(expected, supported("google", "gemini-3.8-flash"))
    }

    @Test
    fun `other text models keep pass-through level without a declared set`() {
        assertNull(supported("vertex", "gemini-3.7-flash"))
    }
}
