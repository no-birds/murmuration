package org.nobirds.murmuration.agent

import java.util.UUID
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutor
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 * Proves the [AgentExecutionIdentity] coroutine-context-element contract that its KDoc
 * describes and that a real executor (`AzureOpenAiClient`) relies on: an identity set via
 * `withContext` at the CALL SITE is visible inside a cold [kotlinx.coroutines.flow.Flow]'s body
 * at the point of `.collect` — not at the point the flow was built — because
 * [PromptExecutor.execute] returns a cold flow whose suspend body runs in the collector's
 * coroutine context (standard cold-flow context preservation).
 */
class AgentExecutionIdentityTest {

    private var capturedPipelineId: UUID? = null
    private var capturedAgentName: String? = null

    // Minimal executor stub mirroring what AzureOpenAiClient does in production: read
    // AgentExecutionIdentity from the ambient coroutine context *inside* the flow body, never
    // from the Prompt argument itself.
    private val identityCapturingExecutor = PromptExecutor { _ ->
        flow {
            val identity = currentCoroutineContext()[AgentExecutionIdentity]
            capturedPipelineId = identity?.pipelineId
            capturedAgentName = identity?.agentName
            emit(PromptExecutionEvent.Completed())
        }
    }

    private val testLlm = Llm(
        provider = "test", family = "test", name = "test",
        version = "1", stable = true, model = "test",
    )

    private fun dummyPrompt(): Prompt = Prompt(testLlm, settings = null, content = emptyList())

    private fun reset() {
        capturedPipelineId = null
        capturedAgentName = null
    }

    @Test
    fun `identity set via withContext around collect is visible inside the executor's flow body`() = runBlocking {
        reset()
        val pipelineId = UUID.randomUUID()
        withContext(AgentExecutionIdentity(pipelineId, "SomeAgent#a1b2[CallToolAgent]")) {
            identityCapturingExecutor.execute(dummyPrompt()).collect { }
        }
        assertEquals(pipelineId, capturedPipelineId)
        assertEquals("SomeAgent#a1b2[CallToolAgent]", capturedAgentName)
    }

    @Test
    fun `no identity in context leaves it null inside the executor's flow body`() = runBlocking {
        reset()
        identityCapturingExecutor.execute(dummyPrompt()).collect { }
        assertNull(capturedPipelineId)
        assertNull(capturedAgentName)
    }

    @Test
    fun `wrapping prompt construction alone, not the collect, does NOT make the identity visible`() = runBlocking {
        reset()
        // The cold flow is built INSIDE the withContext block, but collected OUTSIDE it — the
        // exact mistake [AgentExecutionIdentity]'s KDoc warns against. `flow { }` bodies don't
        // run until collected, so the identity set here must NOT leak into the eventual
        // collection.
        val flow = withContext(AgentExecutionIdentity(UUID.randomUUID(), "ShouldNotLeak")) {
            identityCapturingExecutor.execute(dummyPrompt())
        }
        flow.collect { }
        assertNull(capturedPipelineId)
        assertNull(capturedAgentName)
    }

    @Test
    fun `two sequential collects under different identities capture independently`() = runBlocking {
        reset()
        val pipelineId = UUID.randomUUID()

        withContext(AgentExecutionIdentity(pipelineId, "AgentA#1111[CallToolAgent]")) {
            identityCapturingExecutor.execute(dummyPrompt()).collect { }
        }
        assertEquals(pipelineId, capturedPipelineId)
        assertEquals("AgentA#1111[CallToolAgent]", capturedAgentName)

        withContext(AgentExecutionIdentity(pipelineId, "AgentB#2222[CallToolAgent]")) {
            identityCapturingExecutor.execute(dummyPrompt()).collect { }
        }
        assertEquals("AgentB#2222[CallToolAgent]", capturedAgentName)
    }
}
