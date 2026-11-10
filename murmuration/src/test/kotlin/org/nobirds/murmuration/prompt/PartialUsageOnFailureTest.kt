package org.nobirds.murmuration.prompt

import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertFailsWith

/**
 * [aggregate] must not throw away the statistics already assembled before a stream break: the
 * deltas/parts seen up to the [PromptExecutionEvent.Failed] event are attached to the rethrown
 * [PromptExecutionException] via [PromptExecutionException.partialStats] (widened from a bare
 * token total — so failure statistics rebuilt downstream
 * carry the round-trip's real partial content instead of zero.
 */
class PartialUsageOnFailureTest {

    @Test
    fun `aggregate attaches partial statistics to the failure`() = runBlocking {
        val flow = flowOf<PromptExecutionEvent>(
            PromptExecutionEvent.PromptStarted("test-model", "test-model", emptyList()),
            PromptExecutionEvent.UsageDelta(
                input = InputTokenStatistics(total = 120, cached = 40, cacheWriteShort = 0, cacheWriteLong = 0),
                output = OutputTokenStatistics(total = 30, thinking = 25),
            ),
            PromptExecutionEvent.Failed(
                TransientException(TransientReason.INCOMPLETE_RESPONSE, "stream dropped", null),
            ),
        )

        val ex = assertFailsWith<TransientException> { flow.aggregate() }

        val partial = ex.partialStats
        assertEquals("test-model", partial?.modelName, "modelName from PromptStarted must survive too, not just tokens")
        val usage = partial?.tokens
        assertEquals(120L, usage?.input?.total)
        assertEquals(40L, usage?.input?.cached)
        assertEquals(30L, usage?.output?.total)
        assertEquals(25L, usage?.output?.thinking)
        assertEquals(150L, usage?.totalTokens)
    }

    @Test
    fun `aggregate leaves partial statistics null when nothing arrived before the failure`() = runBlocking {
        val flow = flowOf<PromptExecutionEvent>(
            PromptExecutionEvent.Failed(
                TransientException(TransientReason.EMPTY_STREAM, "no chunks", null),
            ),
        )

        val ex = assertFailsWith<TransientException> { flow.aggregate() }

        assertNull(ex.partialStats, "not even PromptStarted arrived, so partialStats stays null")
    }
}
