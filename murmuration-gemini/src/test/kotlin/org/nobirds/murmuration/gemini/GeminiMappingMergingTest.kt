package org.nobirds.murmuration.gemini

import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.CallIds
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.ToolCall
import org.nobirds.murmuration.prompt.ToolCallPromptPart
import kotlin.test.assertEquals

class GeminiMappingMergingTest {

    @Test
    fun testMergingAndStartingWithUser() {
        val contents = listOf(
            PromptContent(PromptContentType.MODEL, listOf(TextPromptPart("Model message 1"))),
            PromptContent(PromptContentType.MODEL, listOf(TextPromptPart("Model message 2"))),
            PromptContent(PromptContentType.USER, listOf(TextPromptPart("User message 1"))),
            PromptContent(PromptContentType.USER, listOf(TextPromptPart("User message 2"))),
            PromptContent(PromptContentType.MODEL, listOf(TextPromptPart("Model message 3")))
        )

        val genaiContents = GeminiHistoryConverter.EMPTY.toGenaiContents(contents)

        // NO LONGER MERGING!
        assertEquals(5, genaiContents.size)

        assertEquals("model", genaiContents[0].role().get())
        assertEquals("Model message 1", genaiContents[0].parts().get()[0].text().get())

        assertEquals("model", genaiContents[1].role().get())
        assertEquals("Model message 2", genaiContents[1].parts().get()[0].text().get())

        assertEquals("user", genaiContents[2].role().get())
        assertEquals("User message 1", genaiContents[2].parts().get()[0].text().get())

        assertEquals("user", genaiContents[3].role().get())
        assertEquals("User message 2", genaiContents[3].parts().get()[0].text().get())

        assertEquals("model", genaiContents[4].role().get())
        assertEquals("Model message 3", genaiContents[4].parts().get()[0].text().get())
    }

    @Test
    fun testSystemInstructionHasNoRole() {
        val content = PromptContent(PromptContentType.SYSTEM, listOf(TextPromptPart("System instruction")))
        val genaiContent = GeminiHistoryConverter.EMPTY.toGenaiContent(content)
        
        assertEquals(false, genaiContent.role().isPresent)
        assertEquals(1, genaiContent.parts().get().size)
        assertEquals("System instruction", genaiContent.parts().get()[0].text().get())
    }

    @Test
    fun `model turn in arrival order goes to the wire with calls last`() {
        fun call(n: String) = ToolCallPromptPart(ToolCall(n, emptyMap(), null, CallIds.mint()))
        val turn = PromptContent(
            PromptContentType.MODEL,
            listOf(TextPromptPart("t1"), call("c1"), TextPromptPart("t2"), call("c2")),
        )

        val parts = GeminiHistoryConverter.EMPTY.toGenaiContents(listOf(turn)).single().parts().get()

        assertEquals(
            listOf("t1", "t2", "fn:c1", "fn:c2"),
            parts.map { it.text().orElse(null) ?: "fn:" + it.functionCall().get().name().get() },
        )
    }
}
