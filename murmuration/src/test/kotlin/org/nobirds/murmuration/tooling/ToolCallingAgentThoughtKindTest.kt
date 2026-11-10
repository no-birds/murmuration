package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.prompt.CallIds
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.agent.execute
import org.nobirds.murmuration.agent.executePipeline
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PartKind
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.PromptPart
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.ToolCallPromptPart
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import kotlin.test.assertEquals

/**
 * Guard for the history kind of a thought: a response thought is stored as [PartKind.THOUGHT]
 * (set from the event type, never inferred from a signature) in both history builds -- the
 * tool-call round-trip (`buildContents`) and the terminal text branch (`asContent`).
 */
class ToolCallingAgentThoughtKindTest {

    class PingTool {
        @Instrument("Ping the test tool.")
        fun ping(): String = "pong"
    }

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    /** Compact, order-sensitive rendering of a MODEL part. */
    private fun PromptPart.shape(): String = when (this) {
        is TextPromptPart -> "${kind.name}($text${thoughtSignature?.let { "|$it" } ?: ""})"
        is ToolCallPromptPart -> "CALL(${call.name}${call.thoughtSignature?.let { "|$it" } ?: ""})"
        else -> this::class.simpleName.orEmpty()
    }

    /** Runs the real agent loop: round-trip 1 emits [firstTurn], round-trip 2 captures the prompt. */
    private fun capturedSecondPrompt(firstTurn: List<MessagePart>, secondTurn: List<MessagePart>): Prompt = runBlocking {
        var calls = 0
        var captured: Prompt? = null
        val executor = PromptExecutor { prompt ->
            flow {
                calls++
                emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
                if (calls == 1) {
                    firstTurn.forEach { emit(PartReceived(it)) }
                } else {
                    captured = prompt
                    secondTurn.forEach { emit(PartReceived(it)) }
                }
                emit(Completed(finishReason = "completed"))
            }
        }
        executePipeline {
            val agent = toolCallingAgent<Unit, PingTool>(
                llm = fakeLlm, executor = executor, name = "PartOrderAgent", maxTurns = 3,
            ) { _, _, _ -> user { text("test prompt") } }
            execute(agent, ToolCallingInput(Unit, PingTool()))
        }
        captured ?: error("second round-trip prompt was not captured")
    }

    /**
     * MODEL contents of the captured prompt. The agent keeps appending to the same history, so
     * after the loop ends the list also holds the final text turn (built by `asContent`'s text
     * branch) as the last MODEL content.
     */
    private fun Prompt.modelShapes(): List<List<String>> =
        content.filter { it.type == PromptContentType.MODEL }.map { c -> c.parts.map { it.shape() } }

    @Test
    fun `thought without signature stays a thought`() {
        val prompt = capturedSecondPrompt(
            firstTurn = listOf(
                MessagePart.Thought("unsigned thought", null),
                MessagePart.ToolCall("ping", emptyMap(), null, callId = CallIds.mint()),
            ),
            secondTurn = listOf(MessagePart.Text("done")),
        )
        assertEquals(
            listOf(listOf("THOUGHT(unsigned thought)", "CALL(ping)"), listOf("TEXT(done)")),
            prompt.modelShapes(),
        )
    }

    @Test
    fun `thought in the final text turn keeps its kind and order`() {
        val prompt = capturedSecondPrompt(
            firstTurn = listOf(MessagePart.ToolCall("ping", emptyMap(), null, callId = CallIds.mint())),
            secondTurn = listOf(
                MessagePart.Thought("final thought", "fsig"),
                MessagePart.Text("answer"),
            ),
        )
        assertEquals(
            listOf(listOf("CALL(ping)"), listOf("THOUGHT(final thought|fsig)", "TEXT(answer)")),
            prompt.modelShapes(),
        )
    }

    @Test
    fun `agent history keeps thought call thought call in arrival order`() {
        val prompt = capturedSecondPrompt(
            firstTurn = listOf(
                MessagePart.Thought("t1", "s1"),
                MessagePart.ToolCall("ping", emptyMap(), null, callId = CallIds.mint()),
                MessagePart.Thought("t2", "s2"),
                MessagePart.ToolCall("ping", emptyMap(), null, callId = CallIds.mint()),
            ),
            secondTurn = listOf(MessagePart.Text("done")),
        )
        assertEquals(
            listOf(
                listOf("THOUGHT(t1|s1)", "CALL(ping)", "THOUGHT(t2|s2)", "CALL(ping)"),
                listOf("TEXT(done)"),
            ),
            prompt.modelShapes(),
        )
    }

    @Test
    fun `inline data is not in the agent history`() {
        val prompt = capturedSecondPrompt(
            firstTurn = listOf(
                MessagePart.Text("look"),
                MessagePart.InlineData("image/png", 3L),
                MessagePart.ToolCall("ping", emptyMap(), null, callId = CallIds.mint()),
            ),
            secondTurn = listOf(MessagePart.Text("done")),
        )
        assertEquals(
            listOf(listOf("TEXT(look)", "CALL(ping)"), listOf("TEXT(done)")),
            prompt.modelShapes(),
        )
    }
}
