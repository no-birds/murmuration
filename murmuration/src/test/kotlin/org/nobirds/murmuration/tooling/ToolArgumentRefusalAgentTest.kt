package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.prompt.CallIds
import ch.qos.logback.classic.Level
import ch.qos.logback.classic.Logger
import ch.qos.logback.classic.spi.ILoggingEvent
import ch.qos.logback.core.read.ListAppender
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
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
import org.nobirds.murmuration.prompt.ToolResultPromptPart
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.schema.SchemaInfo
import org.slf4j.LoggerFactory
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * A refused tool argument travels the path of a failing tool, through the real agent loop: the
 * next round-trip carries `Error: … Allowed values: …`, the tool body never runs, the agent goes on,
 * the refusal is logged as WARN without a stack trace, `onToolCallError` fires (without
 * `onBeforeToolCall`). A throwing `onBeforeToolCall` is a tool error as well, not a crash.
 *
 * Canonical pattern: a scripted [PromptExecutor] emits the tool call on round-trip 0 and text on
 * round-trip 1; the test reads the tool result off the captured second prompt.
 */
class ToolArgumentRefusalAgentTest {

    @Serializable
    enum class DiceType {
        @SerialName("D4") D4, @SerialName("D6") D6, @SerialName("D20") D20,
    }

    @Serializable
    data class Roll(@SchemaInfo(description = "dice") val diceType: DiceType)

    class Dice {
        val invocations = AtomicInteger()

        @Instrument("Rolls dice")
        fun roll(@InstrumentParam("rolls") rolls: List<Roll>): String {
            invocations.incrementAndGet()
            return "rolled ${rolls.size}"
        }
    }

    class Recorder(private val throwBefore: Boolean = false) : ToolCallingListener {
        val before = mutableListOf<String>()
        val errors = mutableListOf<Throwable>()

        override suspend fun onBeforeToolCall(state: AgentState, name: String, args: Map<String, Any?>) {
            before += name
            if (throwBefore) throw IllegalStateException("hook exploded")
        }

        override suspend fun onToolCallError(state: AgentState, name: String, error: Throwable) {
            errors += error
        }
    }

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    private val appender = ListAppender<ILoggingEvent>()
    // The tool loop logs from a file-private executor class: listen on the package logger.
    private val agentLogger = LoggerFactory.getLogger(ToolCallingAgent::class.java.packageName) as Logger

    @BeforeEach
    fun attach() {
        appender.start()
        agentLogger.addAppender(appender)
    }

    @AfterEach
    fun detach() {
        agentLogger.detachAppender(appender)
    }

    private fun run(diceType: String, recorder: Recorder): Pair<List<Prompt>, Dice> = runBlocking {
        val calls = AtomicInteger()
        val captured = mutableListOf<Prompt>()
        val executor = PromptExecutor { prompt ->
            flow {
                val n = calls.getAndIncrement()
                captured += prompt
                emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
                if (n == 0) {
                    val rolls = JsonArray(listOf(
                        buildJsonObject { put("diceType", "d6") },
                        buildJsonObject { put("diceType", diceType) },
                    ))
                    emit(PartReceived(MessagePart.ToolCall("roll", mapOf("rolls" to rolls), callId = CallIds.mint())))
                } else {
                    emit(PartReceived(MessagePart.Text("done")))
                }
                emit(Completed(finishReason = "STOP"))
            }
        }
        val dice = Dice()
        val agent = toolCallingAgent<Unit, Dice>(
            llm = fakeLlm, executor = executor, name = "RefusalAgent", maxTurns = 3,
        ) { _, _, _ -> user { text("roll") } }
        executePipeline {
            register(recorder)
            execute(agent, ToolCallingInput(Unit, dice))
        }
        assertTrue(calls.get() >= 2, "the loop went on to the next round-trip")
        captured to dice
    }

    private fun toolResult(prompts: List<Prompt>): String =
        prompts[1].content.flatMap { it.parts }.filterIsInstance<ToolResultPromptPart>().single().result.let {
            (it as? JsonPrimitive)?.content ?: it.toString()
        }

    @Test
    fun `unknown enum value is refused to the model with the allowed list and the tool is not invoked`() {
        val recorder = Recorder()
        val (prompts, dice) = run("D21", recorder)

        assertEquals(
            "Error: parameter 'rolls[1].diceType': unknown value \"D21\". Allowed values: D4, D6, D20.",
            toolResult(prompts),
        )
        assertEquals(0, dice.invocations.get(), "invoke is not called")
        assertTrue(recorder.before.isEmpty(), "onBeforeToolCall is not fired for refused arguments")
        assertTrue(recorder.errors.single() is ToolArgumentException)

        val refusal = appender.list.single { it.message.contains("refused arguments") }
        assertEquals(Level.WARN, refusal.level)
        assertNull(refusal.throwableProxy, "no stack trace for a refused argument")
        assertTrue(appender.list.none { it.level == Level.ERROR }, "no ERROR for a refused argument")
    }

    @Test
    fun `known value in any case reaches the tool`() {
        val (prompts, dice) = run("d20", Recorder())
        assertEquals("rolled 2", toolResult(prompts))
        assertEquals(1, dice.invocations.get())
    }

    @Test
    fun `throwing onBeforeToolCall becomes a tool error, not a crash`() {
        val recorder = Recorder(throwBefore = true)
        val (prompts, dice) = run("d20", recorder)
        assertEquals("Error: hook exploded", toolResult(prompts))
        assertEquals(0, dice.invocations.get())
        assertTrue(recorder.errors.single() is IllegalStateException)
    }
}
