package org.nobirds.murmuration.prompt

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame

class TurnLayoutTest {

    private fun text(t: String) = TextPromptPart(t)
    private fun call(n: String) = ToolCallPromptPart(ToolCall(n, emptyMap(), null, CallIds.mint()))
    private fun List<PromptPart>.shape() = map { if (it is TextPromptPart) it.text else (it as ToolCallPromptPart).call.name }

    @Test
    fun `non-calls first then calls, each group stable`() {
        val turn = listOf(text("t1"), call("c1"), text("t2"), call("c2"))
        assertEquals(listOf("t1", "t2", "c1", "c2"), turn.callsLast().shape())
    }

    @Test
    fun `idempotent`() {
        val once = listOf(call("c1"), text("t1"), call("c2")).callsLast()
        assertEquals(once.shape(), once.callsLast().shape())
    }

    @Test
    fun `turn without calls is returned as is`() {
        val turn = listOf(text("a"), text("b"))
        assertSame(turn, turn.callsLast())
    }

    @Test
    fun `only model contents are laid out`() {
        val parts = listOf(call("c1"), text("t1"))
        val contents = listOf(PromptContent(PromptContentType.MODEL, parts), PromptContent(PromptContentType.USER, parts))
        val out = contents.withCallsLast()
        assertEquals(listOf("t1", "c1"), out[0].parts.shape())
        assertEquals(listOf("c1", "t1"), out[1].parts.shape())
    }
}
