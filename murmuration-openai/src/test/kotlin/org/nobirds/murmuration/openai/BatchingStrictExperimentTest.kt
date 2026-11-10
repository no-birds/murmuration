package org.nobirds.murmuration.openai

import com.openai.client.OpenAIClient
import com.openai.models.Reasoning
import com.openai.models.ReasoningEffort
import com.openai.models.responses.FunctionTool
import com.openai.models.responses.ResponseCreateParams
import com.openai.models.responses.ResponseIncludable
import com.openai.models.responses.Tool
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.InstrumentDeclaration
import org.nobirds.murmuration.prompt.InstrumentParameter
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.schema.toFunctionSchema
import kotlin.reflect.KType
import kotlin.reflect.typeOf

/**
 * Manual, real-endpoint experiment probing one specific hypothesis about the batching gap
 * observed between gpt-5-mini and gpt-5.4-nano on the same Azure Responses API harness: nano
 * batches ~3.2 independent tool calls per round-trip, mini emits exactly 1 call/RT in ~72% of
 * observed turns. The suspect: `strict: true` on the tool's `parameters` schema (structured
 * outputs regime — every property in `required`, `additionalProperties: false`) may push the
 * model toward one-call-at-a-time decoding, more so for mini than nano.
 *
 * Production (`AzureOpenAiClient.toOpenAiTool`) always sends `strict = false` for tool params —
 * this test does NOT change that. `strict` is hardcoded there (private top-level extension), so
 * this experiment can't just flip a client flag; it duplicates the tool-building + request
 * assembly here with `strict` as a local parameter, going straight through [DirectGptStrategy]
 * (bypassing `AzureOpenAiClient.executeText`) so both regimes can be exercised side by side.
 *
 * No batching assertions — the point is to eyeball the tool_call count per run, not to gate CI
 * on a still-open hypothesis. Every run is printed as `strict=<bool> rep=<n> tool_calls=<count>
 * names=[...]`.
 *
 * Manual: requires real Azure OpenAI endpoint + az login; costs money (12 calls total: 2 strict
 * regimes x 3 reps, ~1-2K tokens each — copies of the existing manual-test cost profile).
 * Run explicitly with approval only — never from CI.
 */
@Disabled("Manual: paid, ~10 calls of ~1-2K tokens each, pennies; run only with approval")
class BatchingStrictExperimentTest {

    private val endpoint = System.getenv("AZURE_OPENAI_ENDPOINT") ?: ""
    // gpt-5-mini is the model under test — this is exactly the deployment the 1-call/RT 72%
    // figure was observed against; not overridable via env like AzureRealModelIntegrationTest's
    // AZURE_OPENAI_TEST_DEPLOYMENT because the experiment is specifically about mini.
    private val deployment = System.getenv("AZURE_OPENAI_TEST_DEPLOYMENT") ?: "gpt-5-mini"
    private val repsPerRegime = 3

    private fun sdkClient(): OpenAIClient = azureOpenAiClient(endpoint, azureBearerTokenSupplier())

    private fun testLlm(): Llm = Llm("azure", "gpt", "Test", "1", true, deployment)

    /** Hand-written instrument: `updateHp(name: String, hp: Int)` — two independent scalar params,
     * no relation between them, so three calls (A/B/C) are three fully independent invocations
     * with nothing forcing sequential dependency — the shape the batching hypothesis cares about. */
    private class UpdateHpTool : InstrumentDeclaration<String> {
        override val obj: Any get() = this
        override val name = "update_hp"
        override val description = "Sets a character's current HP to an absolute value"
        override val parameters: List<InstrumentParameter> = listOf(
            InstrumentParameter(name = "name", type = typeOf<String>(), description = "Character name"),
            InstrumentParameter(name = "hp", type = typeOf<Int>(), description = "New HP value"),
        )
        override val returnType: KType = typeOf<String>()
        override suspend fun invoke(args: Map<String, Any?>): String = "ok"
    }

    /**
     * Duplicate of `AzureOpenAiClient.toOpenAiTool()` (private in the main source set) with
     * `strict` exposed as a parameter instead of hardcoded `false` — the whole point of this
     * file. Do NOT promote this back into main; production intentionally pins `strict = false`
     * for tool parameters (see `JsonSchemaToOpenAi.kt` doc comment on the two OpenAI regimes).
     */
    private fun InstrumentDeclaration<*>.toOpenAiToolExperimental(strict: Boolean): FunctionTool {
        val schema = toFunctionSchema()
        val parameters = FunctionTool.Parameters.builder()
            .putAllAdditionalProperties(schema.parameters.toOpenAiJson(strict = strict).toJsonValueMap())
            .build()
        return FunctionTool.builder()
            .name(schema.name)
            .description(schema.description)
            .parameters(parameters)
            .strict(strict)
            .build()
    }

    /** Mirrors the request shape `AzureOpenAiClient.executeText` builds, minus the parts that
     * don't matter for this experiment (typed output, maxOutputTokens cap). Reasoning is pinned
     * to LOW — same tier used by the implementer agents in production. */
    private fun buildParams(
        input: List<com.openai.models.responses.ResponseInputItem>,
        tool: UpdateHpTool,
        strict: Boolean,
    ): ResponseCreateParams =
        ResponseCreateParams.builder()
            .model(deployment)
            .inputOfResponse(input)
            .store(false)
            .addInclude(ResponseIncludable.REASONING_ENCRYPTED_CONTENT)
            .reasoning(Reasoning.builder().summary(Reasoning.Summary.AUTO).effort(ReasoningEffort.LOW).build())
            .parallelToolCalls(true)
            .tools(listOf(Tool.ofFunction(tool.toOpenAiToolExperimental(strict))))
            .build()

    private val prompt = "Update the HP of three characters via the update_hp tool: A=10, B=12, C=7. " +
        "Call the tool separately for each character."

    /** Runs one request under [strict] and returns the number of tool_call parts that arrived
     * in the single response, plus the called names (order as emitted) for the log line. */
    private fun runOnce(client: OpenAIClient, tool: UpdateHpTool, strict: Boolean): Pair<Int, List<String>> = runBlocking {
        val promptObj = buildPrompt {
            llm = testLlm()
            user { text(prompt) }
        }
        val input = promptObj.content.toResponseInput()
        val requestMessages = promptObj.content.toMessages()
        val params = buildParams(input, tool, strict)

        val events = DirectGptStrategy(client).execute(promptObj, params, requestMessages, deployment).toList()
        val calls = events
            .filterIsInstance<PromptExecutionEvent.PartReceived>()
            .mapNotNull { it.part as? MessagePart.ToolCall }
        calls.size to calls.map { it.name + it.args }
    }

    @Test
    fun `strict=true vs strict=false batching on gpt-5-mini, 3 reps each`() {
        val client = sdkClient()
        val tool = UpdateHpTool()

        val results = mutableListOf<Triple<Boolean, Int, Int>>() // (strict, rep, toolCallCount)
        for (strict in listOf(true, false)) {
            repeat(repsPerRegime) { rep ->
                val (count, names) = runOnce(client, tool, strict)
                println("strict=$strict rep=${rep + 1} tool_calls=$count names=$names")
                results += Triple(strict, rep + 1, count)
            }
        }

        println("=== summary ===")
        for (strict in listOf(true, false)) {
            val counts = results.filter { it.first == strict }.map { it.third }
            println("strict=$strict counts=$counts avg=${counts.average()}")
        }
    }
}
