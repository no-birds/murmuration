package org.nobirds.murmuration.openai

import org.nobirds.murmuration.prompt.CallIds
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.agent.execute
import org.nobirds.murmuration.agent.executePipeline
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.tooling.Instrument
import org.nobirds.murmuration.tooling.ToolCallingInput
import org.nobirds.murmuration.tooling.toolCallingAgent
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Full-circle regression test for reasoning replay: a model turn that returned a reasoning item
 * (thought signature) plus a function call must, after the CORE agent loop appends it to the
 * conversation, produce a next-round-trip input where `ResponseInputItem.ofReasoning` (with the
 * original encrypted content) immediately precedes the paired function call.
 *
 * This drives the REAL [org.nobirds.murmuration.tooling.ToolCallingAgent] history path (event
 * collection → TurnResult → buildContents → history prompt), not a hand-built history — exactly
 * the path where a silent drop would disappear from unit tests that construct parts manually.
 */
class ReasoningReplayRoundTripTest {

    class PingTool {
        @Instrument("Ping the test tool.")
        fun ping(): String = "pong"
    }

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    @Test
    fun `reasoning signature survives the core agent loop and replays before its function call`() = runBlocking {
        val signature = encodeReasoningSignature("rs_full_circle", "ENCRYPTED_BLOB")
        val calls = AtomicInteger()
        var secondPrompt: Prompt? = null

        // RT1: a tool call carrying the reasoning signature — exactly what DirectGptStrategy /
        // StreamingGptStrategy emit for a Responses reasoning + function_call turn.
        // RT2: capture the prompt the agent built, then finish with plain text.
        val executor = PromptExecutor { prompt ->
            flow {
                val n = calls.incrementAndGet()
                emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
                if (n == 1) {
                    emit(PartReceived(MessagePart.ToolCall(name = "ping", args = emptyMap(), thoughtSignature = signature, callId = CallIds.mint())))
                    emit(Completed(finishReason = "completed"))
                } else {
                    secondPrompt = prompt
                    emit(PartReceived(MessagePart.Text("done")))
                    emit(Completed(finishReason = "completed"))
                }
            }
        }

        executePipeline {
            val agent = toolCallingAgent<Unit, PingTool>(
                llm = fakeLlm, executor = executor, name = "ReplayProbeAgent", maxTurns = 2,
            ) { _, _, _ -> user { text("test prompt") } }
            execute(agent, ToolCallingInput(Unit, PingTool()))
        }

        assertEquals(2, calls.get(), "agent must run a second round-trip after the tool call")
        val prompt = secondPrompt ?: error("second round-trip prompt was not captured")

        // 1) The signature must still sit on the MODEL tool-call part the agent appended.
        val modelParts = prompt.content.filter { it.type.name == "MODEL" }.flatMap { it.parts }
        val carried = modelParts.mapNotNull { (it as? org.nobirds.murmuration.prompt.ToolCallPromptPart)?.call?.thoughtSignature } +
            modelParts.filterIsInstance<TextPromptPart>().mapNotNull { it.thoughtSignature }
        assertTrue(
            carried.contains(signature),
            "reasoning signature must survive the agent's history building; found signatures: $carried, " +
                "model parts: ${modelParts.map { it::class.simpleName }}",
        )

        // 2) The provider mapping must replay it as a reasoning item immediately before the call.
        val items = prompt.content.toResponseInput()
        val reasoningIdx = items.indexOfFirst { it.isReasoning() }
        assertTrue(reasoningIdx >= 0, "next-round-trip input must contain a reasoning item; items: ${items.map { summarize(it) }}")
        val reasoning = items[reasoningIdx].reasoning().get()
        assertEquals("rs_full_circle", reasoning.id())
        assertEquals("ENCRYPTED_BLOB", reasoning.encryptedContent().get())
        assertTrue(
            items[reasoningIdx + 1].isFunctionCall(),
            "reasoning item must immediately precede its paired function call; items: ${items.map { summarize(it) }}",
        )
        val call = items[reasoningIdx + 1].functionCall().get()
        assertEquals("ping", call.name())
    }

    @Test
    fun `final text turn keeps its reasoning signature on the model text part`() = runBlocking {
        // A reasoning + text (no tool calls) turn — the terminal case. The loop ends, but the
        // signature must not corrupt the assistant text if the history is ever replayed.
        val signature = encodeReasoningSignature("rs_final", "ENC2")
        val history = listOf(
            org.nobirds.murmuration.prompt.PromptContent(
                org.nobirds.murmuration.prompt.PromptContentType.MODEL,
                listOf(
                    TextPromptPart("", signature),
                    TextPromptPart("The answer is 4."),
                ),
            ),
        )
        val items = history.toResponseInput()
        // reasoning replays first, assistant text follows — no orphaned reasoning (the message
        // item is its required following item).
        assertEquals(2, items.size)
        assertTrue(items[0].isReasoning())
        assertTrue(items[1].isEasyInputMessage())
        assertEquals("The answer is 4.", items[1].easyInputMessage().get().content().asTextInput())
    }

    private fun summarize(item: com.openai.models.responses.ResponseInputItem): String = when {
        item.isReasoning() -> "reasoning(${item.reasoning().get().id()})"
        item.isFunctionCall() -> "function_call(${item.functionCall().get().name()})"
        item.isFunctionCallOutput() -> "function_call_output"
        item.isEasyInputMessage() -> "message(${item.easyInputMessage().get().role()})"
        else -> "other"
    }
}
