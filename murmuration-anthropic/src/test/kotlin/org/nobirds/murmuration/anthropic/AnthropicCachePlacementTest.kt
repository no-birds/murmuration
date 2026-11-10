package org.nobirds.murmuration.anthropic

import com.anthropic.models.messages.CacheControlEphemeral
import com.anthropic.models.messages.Tool
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.agent.execute
import org.nobirds.murmuration.agent.executePipeline
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.llm.LlmProviders
import org.nobirds.murmuration.prompt.CallIds
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PIPELINE
import org.nobirds.murmuration.prompt.PartKind
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptCacheStrategy
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.withReplaceableSectionPruning
import org.nobirds.murmuration.prompt.PromptPart
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.ToolsPromptSettings
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.schema.FunctionSchema
import org.nobirds.murmuration.schema.JsonSchemaBuilder.Kind
import org.nobirds.murmuration.schema.jsonSchema
import org.nobirds.murmuration.schema.toFunctionSchema
import org.nobirds.murmuration.tooling.Instrument
import org.nobirds.murmuration.tooling.ToolCallingInput
import org.nobirds.murmuration.tooling.toolCallingAgent
import java.util.Optional
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.hours
import kotlin.time.Duration.Companion.minutes

/**
 * Cache points on an assembled request. [pointsOf] is the canonical way to read the result: one entry per point in API
 * order (`tool:<name>`, `system:<i>`, `msg:<message>.<block>`) with its TTL class.
 */
class AnthropicCachePlacementTest {

    class PingTool {
        @Instrument("Ping the test tool.")
        fun ping(): String = "pong"
    }

    private val fakeLlm = Llm(provider = LlmProviders.ANTHROPIC, family = "test", name = "test", version = "1", stable = true, model = "test-model")
    private val model = AnthropicLlms.findModel("anthropic", "claude-sonnet-5-5")
    private val pingTool = FunctionSchema("ping", "Ping.", jsonSchema { type(Kind.OBJECT) })

    private fun ttlName(control: CacheControlEphemeral) =
        if (control.ttl().get() == CacheControlEphemeral.Ttl.TTL_1H) "1h" else "5m"

    /** Cache points of [request] in API order, e.g. `system:0=5m`, `msg:2.0=5m`. */
    private fun pointsOf(request: CachedRequest): List<String> = buildList {
        request.tools.forEach { tool -> tool.cacheControl().ifPresent { add("tool:${tool.name()}=${ttlName(it)}") } }
        request.system.forEachIndexed { i, block -> block.cacheControl().ifPresent { add("system:$i=${ttlName(it)}") } }
        request.messages.forEachIndexed { m, message ->
            message.content().asBlockParams().forEachIndexed { b, block ->
                val control = when {
                    block.isText() -> block.asText().cacheControl()
                    block.isToolUse() -> block.asToolUse().cacheControl()
                    block.isToolResult() -> block.asToolResult().cacheControl()
                    block.isImage() -> block.asImage().cacheControl()
                    else -> Optional.empty()
                }
                control.ifPresent { add("msg:$m.$b=${ttlName(it)}") }
            }
        }
    }

    private fun materialize(prompt: Prompt, tools: List<Tool> = emptyList()): CachedRequest =
        prompt.toAnthropicMessages().withCachePoints(tools)

    private fun content(type: PromptContentType, vararg parts: PromptPart) = PromptContent(type, parts.toList())

    private fun text(text: String, strategy: PromptCacheStrategy) = TextPromptPart(text, cacheStrategy = strategy)

    private val five = PromptCacheStrategy.ttl(5.minutes)
    private val hour = PromptCacheStrategy.ttl(1.hours)
    private val none = PromptCacheStrategy.NO_CACHE

    // ---- assembled multi-round-trip request (agent + fake MARKS executor) ----

    /** The third round-trip of an agent run on an Anthropic-provider model, the executor decorated with replaceable-section pruning [pruning]. */
    private fun thirdRequest(pruning: Boolean): CachedRequest = runBlocking {
        var calls = 0
        var captured: Prompt? = null
        val executor = PromptExecutor { prompt ->
            flow {
                calls++
                emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
                if (calls < 3) {
                    emit(PartReceived(MessagePart.ToolCall("ping", emptyMap(), null, callId = CallIds.mint())))
                } else {
                    captured = prompt
                    emit(PartReceived(MessagePart.Text("done")))
                }
                emit(Completed(finishReason = "completed"))
            }
        }.withReplaceableSectionPruning(pruning)
        executePipeline {
            val agent = toolCallingAgent<Unit, PingTool>(
                llm = fakeLlm, executor = executor, name = "PlacementAgent", maxTurns = 4,
            ) { _, _, _ -> user { text("stable author context", cacheStrategy = PIPELINE) } }
            execute(agent, ToolCallingInput(Unit, PingTool()))
        }
        val prompt = captured ?: error("third round-trip prompt was not captured")
        val tools = (prompt.settings as ToolsPromptSettings).tools
            .flatMap { it.instruments }.map { it.toFunctionSchema().toAnthropicTool(prompt) }
        materialize(prompt, tools)
    }

    /** On MARKS the agent marks its phase blocks like its history (no holes), so the sliding point reaches the end of the history. */
    @Test
    fun `on a MARKS model the cache point stands at the end of the history on the third round-trip`() {
        val request = thirdRequest(pruning = false)

        val points = pointsOf(request)
        val lastMessage = request.messages.lastIndex
        val lastBlock = request.messages.last().content().asBlockParams().lastIndex
        assertEquals("msg:$lastMessage.$lastBlock=5m", points.last(), "points: $points")
        assertTrue(points.all { it.endsWith("=5m") }, "one TTL class: $points")
    }

    @Test
    fun `with pruning enabled a MARKS model is not pruned, the sliding point is the same`() {
        val request = thirdRequest(pruning = true)

        assertEquals(pointsOf(thirdRequest(pruning = false)), pointsOf(request))
        assertEquals("stable author context", request.messages[0].content().asBlockParams()[0].asText().text())
        assertTrue(request.messages.size > 3, "the agent's history is still there")
    }

    // ---- hand-assembled requests ----

    @Test
    fun `a NO_CACHE hole stops the prefix, points stand only before it`() {
        val prompt = Prompt(
            model, null,
            listOf(
                content(PromptContentType.SYSTEM, text("rules", PromptCacheStrategy.PIPELINE)),
                content(PromptContentType.USER, text("stable user", five)),
                content(PromptContentType.MODEL, text("volatile", none)),
                content(PromptContentType.USER, text("stable again", five)),
                content(PromptContentType.MODEL, text("stable model", five)),
            ),
        )

        val points = pointsOf(materialize(prompt, listOf(pingTool.toAnthropicTool(prompt))))

        assertEquals(listOf("tool:ping=5m", "system:0=5m", "msg:0.0=5m"), points)
    }

    @Test
    fun `a hole in the very first part leaves no point at all, even with tools`() {
        val prompt = Prompt(
            model, null,
            listOf(
                content(PromptContentType.SYSTEM, text("volatile first", none), text("stable later", five)),
                content(PromptContentType.USER, text("go", five)),
            ),
        )

        // Tools alone are stable but no part of the stable prefix was marked: no write is paid.
        assertEquals(emptyList(), pointsOf(materialize(prompt, listOf(pingTool.toAnthropicTool(prompt)))))
    }

    @Test
    fun `more candidates than the API allows give exactly four points chosen by the rule`() {
        val history = (0 until 50).map { i ->
            content(if (i % 2 == 0) PromptContentType.USER else PromptContentType.MODEL, text("turn $i", five))
        }
        val prompt = Prompt(model, null, listOf(content(PromptContentType.SYSTEM, text("rules", five))) + history)

        val points = pointsOf(materialize(prompt, listOf(pingTool.toAnthropicTool(prompt))))

        assertEquals(MAX_POINTS, points.size, "points: $points")
        assertEquals("tool:ping=5m", points[0])
        assertEquals("system:0=5m", points[1])
        assertEquals("msg:49.0=5m", points.last())
        // The fourth one is the intermediate point that keeps the lookback gap within the API's window.
        val middle = points[2].removePrefix("msg:").substringBefore('.').toInt()
        assertTrue(middle in 1..48, "points: $points")
    }

    @Test
    fun `a mix of 5 minute and 1 hour wishes becomes one 1 hour class`() {
        val prompt = Prompt(
            model, null,
            listOf(
                content(PromptContentType.SYSTEM, text("rules", PromptCacheStrategy.PIPELINE)),
                content(PromptContentType.USER, text("between turns", hour)),
                content(PromptContentType.MODEL, text("history", five)),
            ),
        )

        val points = pointsOf(materialize(prompt))

        assertTrue(points.size >= 2, "points: $points")
        assertTrue(points.all { it.endsWith("=1h") }, "points: $points")
    }

    @Test
    fun `PIPELINE and 5 minute wishes stay 5 minutes, a 1 hour wish behind the hole does not count`() {
        val prompt = Prompt(
            model, null,
            listOf(
                content(PromptContentType.SYSTEM, text("rules", PromptCacheStrategy.PIPELINE)),
                content(PromptContentType.USER, text("stable", five)),
                content(PromptContentType.MODEL, text("volatile", none)),
                content(PromptContentType.USER, text("long wish after the hole", hour)),
            ),
        )

        val points = pointsOf(materialize(prompt))

        assertTrue(points.isNotEmpty())
        assertTrue(points.all { it.endsWith("=5m") }, "points: $points")
    }

    @Test
    fun `a tiny prefix is not an error, the point is still placed`() {
        val prompt = Prompt(model, null, listOf(content(PromptContentType.USER, text("hi", five))))

        assertEquals(listOf("msg:0.0=5m"), pointsOf(materialize(prompt)))
    }

    @Test
    fun `the point steps back over a block that cannot carry cache_control`() {
        val prompt = Prompt(
            model, null,
            listOf(
                content(PromptContentType.USER, text("go", five)),
                content(
                    PromptContentType.MODEL,
                    text("answer", five),
                    TextPromptPart(
                        "thinking", thoughtSignature = AnthropicThoughtCarrier.encodeSignature("SIG"),
                        cacheStrategy = five, kind = PartKind.THOUGHT,
                    ),
                ),
            ),
        )

        assertEquals(listOf("msg:1.0=5m"), pointsOf(materialize(prompt)))
    }

    @Test
    fun `the rebuilt request keeps the content, only cache_control is added`() {
        val prompt = Prompt(
            model, null,
            listOf(content(PromptContentType.SYSTEM, text("rules", five)), content(PromptContentType.USER, text("go", five))),
        )
        val plain = prompt.toAnthropicMessages()

        val cached = plain.withCachePoints(emptyList())

        assertEquals(plain.system.map { it.text() }, cached.system.map { it.text() })
        assertEquals(
            plain.messages.map { m -> m.content().asBlockParams().map { it.asText().text() } },
            cached.messages.map { m -> m.content().asBlockParams().map { it.asText().text() } },
        )
        assertNotNull(cached.system[0].cacheControl().orElse(null))
    }

    @Test
    fun `the placement of tools, system and a book is one 5 minute class`() {
        val prompt = Prompt(
            model, null,
            listOf(content(PromptContentType.SYSTEM, text("rules", five)), content(PromptContentType.USER, text("book", five))),
        )

        assertEquals(listOf("tool:ping=5m", "system:0=5m", "msg:0.0=5m"), pointsOf(materialize(prompt, listOf(pingTool.toAnthropicTool(prompt)))))
    }

    @Test
    fun `all points of a request share one TTL class, in any combination of wishes`() {
        val wishes = listOf(five, hour, PromptCacheStrategy.PIPELINE)
        for (withTools in listOf(false, true)) for (tail in wishes) for (mid in wishes) {
            val prompt = Prompt(
                model, null,
                listOf(content(PromptContentType.SYSTEM, text("rules", mid)), content(PromptContentType.USER, text("tail", tail))),
            )
            val tools = if (withTools) listOf(pingTool.toAnthropicTool(prompt)) else emptyList()

            val ttls = pointsOf(materialize(prompt, tools)).map { it.substringAfter('=') }.distinct()

            assertTrue(ttls.size <= 1, "tools=$withTools tail=$tail mid=$mid: $ttls")
        }
    }
}
