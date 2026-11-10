package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.prompt.CallIds
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.agent.execute
import org.nobirds.murmuration.agent.executePipeline
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.InstrumentDeclaration
import org.nobirds.murmuration.prompt.InstrumentParameter
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.schema.toFunctionSchema
import java.util.concurrent.ConcurrentHashMap
import kotlin.reflect.KType
import kotlin.reflect.typeOf
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 * The tool name has ONE rule ([instrumentName]) and ONE carrier into the tool body
 * ([CurrentToolCall] / [currentToolName]). Canonical example for the pair: a real
 * [ToolCallingAgent] gets one model batch with four calls, every tool body reports the name it
 * sees, and the test compares it with the schema name and with the name computed from a
 * function / property reference.
 */
class CurrentToolCallTest {

    /** Property tool: the declaration owns its name, the property carries the annotation. */
    class NamedPropertyTool(override val name: String, private val seen: MutableMap<String, String?>) :
        InstrumentDeclaration<Any?> {
        override val obj: Any get() = this
        override val description: String = "property tool"
        override val parameters: List<InstrumentParameter> = emptyList()
        override val returnType: KType = typeOf<Any?>()
        override suspend fun invoke(args: Map<String, Any?>): Any? {
            seen[name] = currentToolName()
            return "ok"
        }
    }

    class NamedTools {
        /** Tool name (as the model calls it) -> name the body saw in its own coroutine context. */
        val seen: MutableMap<String, String?> = ConcurrentHashMap()

        @Instrument("Plain function: name = function name")
        suspend fun plain(@InstrumentParam("Any value") value: String): String {
            seen["plain"] = currentToolName()
            return "ok: $value"
        }

        /** Plain (non-suspend) body: the only reader it has is the thread-bound one. */
        @Instrument("Plain non-suspend function")
        fun nonSuspend(@InstrumentParam("Any value") value: String): String {
            seen["nonSuspend"] = currentToolNameOnThread()
            return "ok: $value"
        }

        @Instrument("Renamed function", name = "renamed_tool")
        suspend fun kotlinNameIsNotTheToolName(@InstrumentParam("Any value") value: String): String {
            seen["renamed_tool"] = currentToolName()
            return "ok: $value"
        }

        @Instrument("Property tool: name = property name")
        val propTool: InstrumentDeclaration<Any?> = NamedPropertyTool("propTool", seen)

        @Instrument("Renamed property tool", name = "renamed_prop")
        val kotlinPropertyName: InstrumentDeclaration<Any?> = NamedPropertyTool("renamed_prop", seen)
    }

    private val fakeLlm = Llm(
        provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model",
    )

    private fun oneBatch(names: List<String>, captured: MutableList<Prompt>): PromptExecutor = PromptExecutor { prompt ->
        flow {
            val turn = captured.size
            captured += prompt
            emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
            if (turn == 0) {
                names.forEach { name ->
                    val args = if (name.endsWith("Tool") || name.endsWith("prop")) emptyMap()
                    else mapOf<String, JsonElement>("value" to JsonPrimitive(name))
                    emit(PartReceived(MessagePart.ToolCall(name, args, callId = CallIds.mint())))
                }
            } else {
                emit(PartReceived(MessagePart.Text("done")))
            }
            emit(Completed(finishReason = "STOP"))
        }
    }

    @Test
    fun `every call of one batch sees its own name, equal to the schema name and to the reference name`() = runBlocking {
        val tools = NamedTools()
        val schemaNames = InstrumentFactory().create(tools).instruments.map { it.toFunctionSchema().name }.toSet()
        val referenceNames = setOf(
            NamedTools::plain.instrumentName(),
            NamedTools::nonSuspend.instrumentName(),
            NamedTools::kotlinNameIsNotTheToolName.instrumentName(),
            NamedTools::propTool.instrumentName(),
            NamedTools::kotlinPropertyName.instrumentName(),
        )
        assertEquals(setOf("plain", "nonSuspend", "renamed_tool", "propTool", "renamed_prop"), referenceNames)
        assertEquals(referenceNames, schemaNames, "the name of a reference is the name in the schema")

        val agent = toolCallingAgent<Unit, NamedTools>(
            llm = fakeLlm, executor = oneBatch(referenceNames.toList(), mutableListOf()), name = "NameAgent", maxTurns = 2,
        ) { _, _, _ -> user { text("test prompt") } }
        executePipeline { execute(agent, ToolCallingInput(Unit, tools)) }
        assertNull(currentToolNameOnThread(), "the thread-bound name does not leak out of the call")

        assertEquals(
            referenceNames.associateWith { it as String? }, tools.seen.toMap(),
            "each body (suspend or plain) saw exactly the name under which the model called it",
        )
    }

    @Test
    fun `outside a tool call there is no current name`() = runBlocking {
        assertNull(currentToolName())
        assertNull(currentToolNameOnThread())
    }
}
