package org.nobirds.murmuration.anthropic

import org.nobirds.murmuration.llm.CostCategory
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.llm.LlmCostBreakdown
import org.nobirds.murmuration.llm.LlmCostCalculator
import org.nobirds.murmuration.llm.RoundTripUsage
import org.nobirds.murmuration.prompt.InputTokenStatistics
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.OutputTokenStatistics
import org.nobirds.murmuration.prompt.PartKind
import org.nobirds.murmuration.prompt.PromptCacheStrategy
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.UsageDelta
import org.nobirds.murmuration.prompt.PromptPart
import org.nobirds.murmuration.prompt.ToolCall
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.ThinkingLevel
import org.nobirds.murmuration.prompt.ToolCallPromptPart
import org.nobirds.murmuration.prompt.ToolResultPromptPart
import org.nobirds.murmuration.prompt.ToolsPromptSettings
import org.nobirds.murmuration.prompt.TypedPromptSettings
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.plus
import org.nobirds.murmuration.prompt.system
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.schema.SchemaInfo
import org.nobirds.murmuration.tooling.Instrument
import org.nobirds.murmuration.tooling.buildInstruments
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Paid, real-endpoint tests of [AnthropicPromptExecutor] against the live Claude API (cents per run).
 *
 * They run only when `ANTHROPIC_TEST_API_KEY` is set (otherwise JUnit reports them as skipped);
 * `ANTHROPIC_TEST_WORKSPACE_ID` is optional. The own variable names are deliberate: the test task
 * poisons `ANTHROPIC_API_KEY` & co. so that nothing but explicit configuration can select the auth,
 * and the values reach [AnthropicAuth.ApiKey] explicitly. The key is never printed.
 *
 * Canonical shape: [call] one round-trip -> [Trip] (parts, usage, finish reason), assert on it;
 * [cost] prices a trip through [LlmCostCalculator]. Every test prints a one-line trace to stdout.
 */
@EnabledIfEnvironmentVariable(named = "ANTHROPIC_TEST_API_KEY", matches = ".+")
class AnthropicLiveTest {

    private val haiku = AnthropicLlms.findModel("anthropic", "claude-haiku-5-5")
    private val sonnet = AnthropicLlms.findModel("anthropic", "claude-sonnet-5-5")

    private val executor = AnthropicPromptExecutor(
        AnthropicClientFactory.create(
            AnthropicAuth.ApiKey(
                key = System.getenv("ANTHROPIC_TEST_API_KEY"),
                workspaceId = System.getenv("ANTHROPIC_TEST_WORKSPACE_ID")?.takeIf { it.isNotBlank() },
            ),
        ),
    )

    /** One finished round-trip: what the model produced and what it cost in tokens. */
    private class Trip(val llm: Llm, val events: List<PromptExecutionEvent>) {
        val parts: List<MessagePart> = events.filterIsInstance<PartReceived>().map { it.part }
        val text: String = parts.filterIsInstance<MessagePart.Text>().joinToString("") { it.text }
        val toolCalls: List<MessagePart.ToolCall> = parts.filterIsInstance<MessagePart.ToolCall>()
        val finishReason: String? = events.filterIsInstance<Completed>().single().finishReason
        val input: InputTokenStatistics = events.filterIsInstance<UsageDelta>()
            .fold(InputTokenStatistics.EMPTY) { acc, d -> InputTokenStatistics(
                acc.total + d.input.total, acc.cached + d.input.cached,
                acc.cacheWriteShort + d.input.cacheWriteShort, acc.cacheWriteLong + d.input.cacheWriteLong,
            ) }
        val output: OutputTokenStatistics = events.filterIsInstance<UsageDelta>()
            .fold(OutputTokenStatistics.EMPTY) { acc, d -> OutputTokenStatistics(acc.total + d.output.total, acc.thinking + d.output.thinking) }

        val cost: LlmCostBreakdown
            get() = LlmCostCalculator.roundTrip(RoundTripUsage(input, output.total, output.thinking), llm.pricingProfile)

        fun trace(label: String) = println(
            "LIVE[$label] ${llm.model} finish=$finishReason in=$input out=$output parts=${parts.map { it::class.simpleName }} cost=${cost.totalUsd}",
        )
    }

    private fun call(prompt: Prompt): Trip = Trip(prompt.llm, runBlocking { executor.execute(prompt).toList() })

    private fun settings(maxOutputTokens: Int = 1024, level: ThinkingLevel? = null) =
        ToolsPromptSettings(tools = emptyList(), thinkingLevel = level, maxOutputTokens = maxOutputTokens)

    /** The model's answer as a history message, replayable on the next round-trip (thoughts with their signatures). */
    private fun Trip.asModelContent(): PromptContent = PromptContent(
        PromptContentType.MODEL,
        parts.mapNotNull<MessagePart, PromptPart> { part ->
            when (part) {
                is MessagePart.Text -> TextPromptPart(part.text)
                is MessagePart.Thought -> TextPromptPart(part.text, thoughtSignature = part.signature, kind = PartKind.THOUGHT)
                is MessagePart.ToolCall -> ToolCallPromptPart(ToolCall(part.name, part.args, part.thoughtSignature, part.callId))
                else -> null
            }
        },
    )

    private fun userContent(text: String): PromptContent =
        PromptContent(PromptContentType.USER, listOf(TextPromptPart(text)))

    private fun toolResults(vararg pairs: Pair<MessagePart.ToolCall, String>): PromptContent = PromptContent(
        PromptContentType.TOOL,
        pairs.map { (call, value) -> ToolResultPromptPart(call.name, JsonPrimitive(value), call.callId) },
    )

    class CityTools {
        @Instrument("Current temperature in Paris, degrees Celsius.")
        fun temperature(): String = "21"

        @Instrument("Current population of Paris.")
        fun population(): String = "2100000"
    }

    @Serializable
    data class Verdict(
        @SchemaInfo("true when the number is even") val even: Boolean,
        @SchemaInfo("the number that was asked about") val number: Int,
    )

    // ---- 1. plain answer ----

    @Test
    fun `plain text answer`() {
        val trip = call(buildPrompt {
            llm = haiku
            settings = settings()
            user { text("Reply with exactly one word: pong") }
        })
        trip.trace("plain")
        assertEquals("end_turn", trip.finishReason)
        assertTrue(trip.text.contains("pong", ignoreCase = true), trip.text)
        assertTrue(trip.input.total > 0 && trip.output.total > 0)
    }

    // ---- 2. stream: SSE events, one text part per content block ----

    @Test
    fun `stream arrives as SSE events - started first, one text part per block, usage in two deltas, finish`() {
        val trip = call(buildPrompt {
            llm = haiku
            settings = settings()
            user { text("Count from 1 to 30, separated by commas, nothing else.") }
        })
        trip.trace("stream")
        val events = trip.events
        // Stream start comes from message_start, so it precedes every part and is emitted exactly once.
        assertEquals(1, events.count { it is PromptExecutionEvent.PromptStarted })
        assertTrue(events.first() is PromptExecutionEvent.PromptStarted, "first event: ${events.first()}")
        assertEquals(haiku.model, (events.first() as PromptExecutionEvent.PromptStarted).requestedModelName)
        // The unit of a text part is the content block (text_deltas are concatenated at content_block_stop):
        // a plain answer is a single text block, hence exactly one Text part, after the start event.
        val texts = trip.parts.filterIsInstance<MessagePart.Text>()
        assertEquals(1, texts.size, "one text block -> one Text part, got ${texts.size}")
        assertTrue(events.indexOfFirst { it is PartReceived } > 0)
        assertTrue(texts.single().text.contains("30"), trip.text)
        // Usage comes in at least two deltas: input at message_start, output at message_delta - a non-streaming
        // call would carry it once. Input is on the first delta, output on a later one.
        val usage = events.filterIsInstance<UsageDelta>()
        assertTrue(usage.size >= 2, "expected usage deltas from message_start and message_delta, got ${usage.size}")
        assertTrue(usage.first().input.total > 0 && usage.last().output.total > 0, usage.toString())
        // Finish is the last event and carries the raw stop_reason.
        assertTrue(events.last() is Completed, "last event: ${events.last()}")
        assertEquals("end_turn", trip.finishReason)
    }

    // ---- 3. two parallel tool calls, CallId end to end ----

    @Test
    fun `two parallel tool calls are answered and the second round-trip finishes the turn`() {
        val tools = ToolsPromptSettings(tools = buildInstruments { from(CityTools()) }, maxOutputTokens = 1024)
        val first = buildPrompt {
            llm = haiku
            settings = tools
            user { text("Use BOTH tools (temperature and population) at once, in parallel, then answer in one sentence.") }
        }
        val firstTrip = call(first)
        firstTrip.trace("tools-1")
        assertEquals("tool_use", firstTrip.finishReason)
        assertEquals(setOf("temperature", "population"), firstTrip.toolCalls.map { it.name }.toSet(), "both tools in one answer")
        assertEquals(firstTrip.toolCalls.size, firstTrip.toolCalls.map { it.callId }.toSet().size, "CallIds are distinct")

        val byName = firstTrip.toolCalls.associateBy { it.name }
        val secondTrip = call(
            first + firstTrip.asModelContent() + toolResults(
                // Results go back in the reverse order of the calls: pairing is by CallId.
                byName.getValue("population") to "2100000",
                byName.getValue("temperature") to "21",
            ),
        )
        secondTrip.trace("tools-2")
        assertEquals("end_turn", secondTrip.finishReason)
        assertTrue(secondTrip.toolCalls.isEmpty())
        assertTrue(secondTrip.text.contains("21"), secondTrip.text)
    }

    // ---- 4. structured output ----

    @Test
    fun `structured output follows the schema`() {
        val trip = call(buildPrompt {
            llm = haiku
            settings = TypedPromptSettings(Verdict::class, maxOutputTokens = 512)
            user { text("Is the number 42 even? Answer for the number 42.") }
        })
        trip.trace("typed")
        assertEquals("end_turn", trip.finishReason)
        val verdict = Json.decodeFromString<Verdict>(trip.text.trim())
        assertEquals(Verdict(even = true, number = 42), verdict)
    }

    // ---- 5. cache: write on the first request, read on the second ----

    @Test
    fun `cache - first request writes 5m, second reads, cost matches the manual recount`() {
        // >2048 tokens of stable system text, unique per run so that an earlier run cannot already hold the cache.
        val nonce = System.nanoTime()
        val rules = buildString {
            append("Cache probe $nonce. You are a terse assistant; answer with one short word.\n")
            repeat(450) { i -> append("Rule ${i + 1}: in the realm of probe $nonce the lantern number ${i * 7 + 3} burns for ${i % 11 + 2} nights.\n") }
        }
        fun request() = buildPrompt {
            llm = haiku
            settings = settings(maxOutputTokens = 64)
            system { text(rules, cacheStrategy = PromptCacheStrategy.PIPELINE) }
            user { text("Say: ok") }
        }

        val firstTrip = call(request())
        firstTrip.trace("cache-1")
        val secondTrip = call(request())
        secondTrip.trace("cache-2")

        assertTrue(firstTrip.input.cacheWriteShort > 2048, "first request writes the 5m cache: ${firstTrip.input}")
        assertEquals(0, firstTrip.input.cacheWriteLong)
        assertEquals(0, firstTrip.input.cached)
        assertTrue(secondTrip.input.cached > 2048, "second request reads the cache: ${secondTrip.input}")
        assertEquals(0, secondTrip.input.cacheWriteLong)

        val firstCost = firstTrip.cost
        val secondCost = secondTrip.cost
        assertTrue(firstCost[CostCategory.CACHE_WRITE] > 0.0)
        assertEquals(0.0, firstCost[CostCategory.CACHED_INPUT])
        assertTrue(secondCost[CostCategory.CACHED_INPUT] > 0.0)

        // Manual recount at the Haiku 5.5 base tier (USD per 1M): input 0.10, read 0.01, write 5m 0.125, output 0.50.
        fun manual(t: Trip): Double =
            t.input.uncached * 0.10 / 1e6 + t.input.cached * 0.01 / 1e6 + t.input.cacheWriteShort * 0.125 / 1e6 + t.output.total * 0.50 / 1e6
        assertEquals(manual(firstTrip), firstCost.totalUsd, 1e-12)
        assertEquals(manual(secondTrip), secondCost.totalUsd, 1e-12)
        println("LIVE[cache-cost] first=${firstCost.totalUsd} (write=${firstCost[CostCategory.CACHE_WRITE]}) second=${secondCost.totalUsd} (read=${secondCost[CostCategory.CACHED_INPUT]})")
    }

    // ---- 6. sonnet: a multi-turn tool run, the first exchange cut out, thoughts stay valid ----

    /**
     * Runs exchange A, then B, then cuts A out (the "replaceable exchange" of the prune policy) and asks again.
     * Returns the number of thoughts seen in A and B, so the caller knows whether the risk
     * (a thought in the cut exchange AND in the remaining turn) was really exercised.
     */
    private fun prunedRun(level: ThinkingLevel, task: String): Pair<Int, Int> {
        val tools = ToolsPromptSettings(tools = buildInstruments { from(CityTools()) }, thinkingLevel = level, maxOutputTokens = 4096)
        val base = buildPrompt {
            llm = sonnet
            settings = tools
            system { text("Call ONE tool per answer, never several at once. Think before each tool call.", cacheStrategy = PromptCacheStrategy.PIPELINE) }
            user { text(task) }
        }
        fun answer(call: MessagePart.ToolCall) = if (call.name == "temperature") "21" else "2100000"

        val tripA = call(base)
        tripA.trace("sonnet-$level-A")
        val callA = tripA.toolCalls.first()
        val tripB = call(base + tripA.asModelContent() + toolResults(callA to answer(callA)))
        tripB.trace("sonnet-$level-B")
        assertTrue(tripB.toolCalls.isNotEmpty(), "model continues with the second tool; finish=${tripB.finishReason}")
        val callB = tripB.toolCalls.first()

        // The prune policy cut exchange A: only B's thought + call + result remain after the user turn.
        val tripC = call(base + tripB.asModelContent() + toolResults(callB to answer(callB)))
        tripC.trace("sonnet-$level-C-pruned")
        assertTrue(tripC.finishReason == "end_turn" || tripC.finishReason == "tool_use", "unexpected finish ${tripC.finishReason}")
        return tripA.parts.count { it is MessagePart.Thought } to tripB.parts.count { it is MessagePart.Thought }
    }

    @Test
    fun `sonnet between_tools survives a pruned exchange`() {
        val (a, b) = prunedRun(ThinkingLevel.MINIMAL, "First get the temperature, then get the population, then say both numbers in one sentence.")
        println("LIVE[sonnet-MINIMAL] thoughts A=$a B=$b")
    }

    @Test
    fun `sonnet adaptive thinking with real thoughts survives a pruned exchange`() {
        val (a, b) = prunedRun(
            ThinkingLevel.HIGH,
            "Work out carefully why 17*23 equals 391, then get the temperature. Once you have it, think hard: " +
                "is it prime, and what is it times 37 times 91? Only after that get the population, " +
                "then answer with both results.",
        )
        println("LIVE[sonnet-HIGH] thoughts A=$a B=$b")
        assertTrue(a + b > 0, "no thought produced: the risk of a thought in a cut exchange was not exercised")
    }
}
