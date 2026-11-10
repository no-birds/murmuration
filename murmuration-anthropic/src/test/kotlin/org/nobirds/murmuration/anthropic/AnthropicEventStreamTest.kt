package org.nobirds.murmuration.anthropic

import com.anthropic.core.http.StreamResponse
import com.anthropic.core.jsonMapper
import com.anthropic.errors.AnthropicIoException
import com.anthropic.models.messages.RawMessageStreamEvent
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.prompt.CallId
import org.nobirds.murmuration.prompt.InputTokenStatistics
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.OutputTokenStatistics
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutionEvent.StreamResponded
import org.nobirds.murmuration.prompt.PromptExecutionEvent.UsageDelta
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.UnknownProviderException
import java.io.InterruptedIOException
import java.util.concurrent.atomic.AtomicReference
import java.util.stream.Stream
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.nanoseconds
import kotlin.time.Duration.Companion.seconds

/**
 * Deterministic tests of the stream parser: events are real SDK events deserialized from the wire JSON of the Messages API,
 * the expectation is the exact sequence of core events. This is the canonical shape — feed `wire(...)` strings, compare `run(...)`.
 */
class AnthropicEventStreamTest {

    private fun wire(json: String): RawMessageStreamEvent = jsonMapper().readValue(json, RawMessageStreamEvent::class.java)

    private fun run(vararg events: String): List<PromptExecutionEvent> {
        val parser = AnthropicStreamParser(requestedModelName = "requested")
        return events.flatMap { parser.accept(wire(it)) }
    }

    private fun messageStart(usage: String = """{"input_tokens":10,"output_tokens":1}""") =
        """{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":$usage}}"""

    private fun blockStart(index: Int, block: String) = """{"type":"content_block_start","index":$index,"content_block":$block}"""
    private fun blockDelta(index: Int, delta: String) = """{"type":"content_block_delta","index":$index,"delta":$delta}"""
    private fun blockStop(index: Int) = """{"type":"content_block_stop","index":$index}"""
    private fun messageDelta(stopReason: String, usage: String) =
        """{"type":"message_delta","delta":{"stop_reason":"$stopReason","stop_sequence":null},"usage":$usage}"""

    private val messageStop = """{"type":"message_stop"}"""
    private val responded = StreamResponded(servedModelName = "claude-sonnet-5-5")
    private val started = PromptStarted(modelName = "requested", requestedModelName = "requested", requestMessages = emptyList())
    private val startUsage = UsageDelta(
        input = InputTokenStatistics(total = 10, cached = 0, cacheWriteShort = 0, cacheWriteLong = 0),
        output = OutputTokenStatistics(total = 1, thinking = 0),
    )

    @Test
    fun `text deltas of one block become ONE text part and stop_reason rides on Completed`() {
        val events = run(
            messageStart(),
            blockStart(0, """{"type":"text","text":""}"""),
            blockDelta(0, """{"type":"text_delta","text":"Hel"}"""),
            blockDelta(0, """{"type":"text_delta","text":"lo"}"""),
            blockStop(0),
            messageDelta("end_turn", """{"output_tokens":15}"""),
            messageStop,
        )
        assertEquals(
            listOf(
                responded,
                startUsage,
                PartReceived(MessagePart.Text("Hello")),
                UsageDelta(output = OutputTokenStatistics(total = 14, thinking = 0)),
                Completed(finishReason = "end_turn"),
            ),
            events,
        )
    }

    @Test
    fun `two text blocks with different indexes are two parts and thinking text tool_use keep their order`() {
        val events = run(
            messageStart(),
            blockStart(0, """{"type":"thinking","thinking":"","signature":""}"""),
            blockDelta(0, """{"type":"thinking_delta","thinking":"hm"}"""),
            blockDelta(0, """{"type":"signature_delta","signature":"SIG"}"""),
            blockStop(0),
            blockStart(1, """{"type":"text","text":""}"""),
            blockDelta(1, """{"type":"text_delta","text":"One "}"""),
            blockDelta(1, """{"type":"text_delta","text":"two"}"""),
            blockStop(1),
            blockStart(2, """{"type":"text","text":"Second"}"""),
            blockStop(2),
            blockStart(3, """{"type":"tool_use","id":"toolu_A","name":"roll","input":{}}"""),
            blockStop(3),
            messageDelta("tool_use", """{"output_tokens":15}"""),
            messageStop,
        )
        val parts = events.filterIsInstance<PartReceived>().map { it.part }
        assertEquals(
            listOf(
                MessagePart.Thought("hm", AnthropicThoughtCarrier.encodeSignature("SIG")),
                MessagePart.Text("One two"),
                MessagePart.Text("Second"),
                MessagePart.ToolCall("roll", emptyMap(), callId = CallId("toolu_A")),
            ),
            parts,
        )
    }

    @Test
    fun `thinking with signature becomes one thought carrying the signature`() {
        val events = run(
            messageStart(),
            blockStart(0, """{"type":"thinking","thinking":"","signature":""}"""),
            blockDelta(0, """{"type":"thinking_delta","thinking":"Let me "}"""),
            blockDelta(0, """{"type":"thinking_delta","thinking":"think."}"""),
            blockDelta(0, """{"type":"signature_delta","signature":"EqQBCgIYAhIM"}"""),
            blockStop(0),
            blockStart(1, """{"type":"text","text":""}"""),
            blockDelta(1, """{"type":"text_delta","text":"Done"}"""),
            blockStop(1),
            messageStop,
        )
        assertEquals(
            listOf(
                responded,
                startUsage,
                PartReceived(MessagePart.Thought(text = "Let me think.", signature = AnthropicThoughtCarrier.encodeSignature("EqQBCgIYAhIM"))),
                PartReceived(MessagePart.Text("Done")),
                Completed(finishReason = null),
            ),
            events,
        )
    }

    @Test
    fun `two parallel tool_use blocks assemble their input_json_delta pieces by index`() {
        val events = run(
            messageStart(),
            blockStart(0, """{"type":"tool_use","id":"toolu_A","name":"roll","input":{}}"""),
            blockStart(1, """{"type":"tool_use","id":"toolu_B","name":"look","input":{}}"""),
            blockDelta(0, """{"type":"input_json_delta","partial_json":"{\"dice\":"}"""),
            blockDelta(1, """{"type":"input_json_delta","partial_json":"{\"where\""}"""),
            blockDelta(0, """{"type":"input_json_delta","partial_json":"\"d20\"}"}"""),
            blockDelta(1, """{"type":"input_json_delta","partial_json":":\"north\"}"}"""),
            blockStop(0),
            blockStop(1),
            messageDelta("tool_use", """{"output_tokens":40}"""),
            messageStop,
        )
        assertEquals(
            listOf(
                responded,
                startUsage,
                PartReceived(MessagePart.ToolCall("roll", mapOf("dice" to JsonPrimitive("d20")), callId = CallId("toolu_A"))),
                PartReceived(MessagePart.ToolCall("look", mapOf("where" to JsonPrimitive("north")), callId = CallId("toolu_B"))),
                UsageDelta(output = OutputTokenStatistics(total = 39, thinking = 0)),
                Completed(finishReason = "tool_use"),
            ),
            events,
        )
    }

    @Test
    fun `tool_use without input pieces has empty arguments`() {
        val events = run(
            messageStart(),
            blockStart(0, """{"type":"tool_use","id":"toolu_A","name":"ping","input":{}}"""),
            blockStop(0),
        )
        assertEquals(PartReceived(MessagePart.ToolCall("ping", emptyMap(), callId = CallId("toolu_A"))), events.last())
    }

    @Test
    fun `redacted_thinking becomes an empty thought with the redacted carrier`() {
        val events = run(
            messageStart(),
            blockStart(0, """{"type":"redacted_thinking","data":"OPAQUE=="}"""),
            blockStop(0),
            messageStop,
        )
        assertEquals(
            listOf(
                responded,
                startUsage,
                PartReceived(MessagePart.Thought(text = "", signature = AnthropicThoughtCarrier.encodeRedacted("OPAQUE=="))),
                Completed(finishReason = null),
            ),
            events,
        )
    }

    @Test
    fun `usage maps 5m write 1h write and read into input statistics, thinking is split out of output`() {
        val events = run(
            messageStart(
                """{"input_tokens":10,"cache_read_input_tokens":500,"cache_creation_input_tokens":300,""" +
                    """"cache_creation":{"ephemeral_5m_input_tokens":100,"ephemeral_1h_input_tokens":200},"output_tokens":1}"""
            ),
            messageDelta("end_turn", """{"output_tokens":80,"output_tokens_details":{"thinking_tokens":60}}"""),
        )
        assertEquals(
            UsageDelta(
                input = InputTokenStatistics(total = 810, cached = 500, cacheWriteShort = 100, cacheWriteLong = 200),
                output = OutputTokenStatistics(total = 1, thinking = 0),
            ),
            events[1],
        )
        // cumulative message_delta: the increment over what was already sent; summed with the start it is the final value
        assertEquals(UsageDelta(output = OutputTokenStatistics(total = 19, thinking = 60)), events[2])
        assertEquals(810L, (events[1] as UsageDelta).input.total)
        assertEquals(10L, (events[1] as UsageDelta).input.uncached)
    }

    @Test
    fun `aggregate cache_creation that the breakdown does not explain is a typed non-retryable mapping error`() {
        val e = assertFailsWith<UnknownProviderException> {
            run(
                messageStart(
                    """{"input_tokens":10,"cache_creation_input_tokens":300,""" +
                        """"cache_creation":{"ephemeral_5m_input_tokens":100,"ephemeral_1h_input_tokens":150},"output_tokens":1}"""
                )
            )
        }
        assertEquals(USAGE_MAPPING_ERROR, e.rawClass)
    }

    @Test
    fun `cache write without a breakdown is a mapping error`() {
        val e = assertFailsWith<UnknownProviderException> {
            run(messageStart("""{"input_tokens":10,"cache_creation_input_tokens":300,"output_tokens":1}"""))
        }
        assertEquals(USAGE_MAPPING_ERROR, e.rawClass)
    }

    private fun flowOf(vararg events: String) = anthropicEventFlow(
        openStream = {
            object : StreamResponse<RawMessageStreamEvent> {
                override fun stream(): Stream<RawMessageStreamEvent> = Stream.of(*events.map(::wire).toTypedArray())
                override fun close() = Unit
            }
        },
        requestMessages = emptyList(),
        requestedModelName = "requested",
        idleTimeout = 5.seconds,
        overallTimeout = 30.seconds,
    )

    @Test
    fun `flow emits the full event sequence and a stream without message_stop is an incomplete response`() = runBlocking {
        val complete = flowOf(messageStart(), blockStart(0, """{"type":"text","text":"Hi"}"""), blockStop(0), messageStop).toList()
        assertEquals(listOf(started, responded, startUsage, PartReceived(MessagePart.Text("Hi")), Completed(finishReason = null)), complete)

        val e = assertFailsWith<TransientException> { flowOf(messageStart()).toList() }
        assertEquals(TransientReason.INCOMPLETE_RESPONSE, e.reason)

        val empty = assertFailsWith<TransientException> { flowOf().toList() }
        assertEquals(TransientReason.EMPTY_STREAM, empty.reason)
    }

    /**
     * A stream whose `next()` blocks like okhttp does: an interrupt (cancellation) comes out as `InterruptedIOException`
     * wrapped into the SDK's `AnthropicIoException("Stream failed")`, never as `InterruptedException`.
     */
    private fun silentFlow(idleTimeout: Duration?, silence: Duration = 60.seconds, overallTimeout: Duration = 30.seconds) = anthropicEventFlow(
        openStream = {
            object : StreamResponse<RawMessageStreamEvent> {
                override fun stream(): Stream<RawMessageStreamEvent> = Stream.generate<RawMessageStreamEvent> {
                    try {
                        Thread.sleep(silence.inWholeMilliseconds)
                        wire(messageStop)
                    } catch (e: InterruptedException) {
                        throw AnthropicIoException("Stream failed", InterruptedIOException("interrupted"))
                    }
                }.limit(1)

                override fun close() = Unit
            }
        },
        requestMessages = emptyList(),
        requestedModelName = "requested",
        idleTimeout = idleTimeout,
        overallTimeout = overallTimeout,
    )

    @Test
    fun `our own idle timeout is NETWORK_TIMEOUT, not the I-O error the interrupt turns into`() = runBlocking {
        val e = assertFailsWith<TransientException> { silentFlow(idleTimeout = 200.milliseconds).toList() }
        assertEquals(TransientReason.NETWORK_TIMEOUT, e.reason)
    }

    @Test
    fun `a hung read with the idle bound off is cut by the overall deadline as NETWORK_TIMEOUT`() = runBlocking {
        val started = System.nanoTime()
        val e = assertFailsWith<TransientException> {
            silentFlow(idleTimeout = null, overallTimeout = 300.milliseconds).toList()
        }
        assertEquals(TransientReason.NETWORK_TIMEOUT, e.reason)
        assertTrue("overall" in e.message.orEmpty(), e.message)
        assertTrue((System.nanoTime() - started).nanoseconds < 10.seconds, "the read must be cut near the deadline, not after the 60s silence")
    }

    @Test
    fun `without an idle bound a long silence is not a failure`() = runBlocking {
        val events = silentFlow(idleTimeout = null, silence = 600.milliseconds).toList()
        assertEquals(listOf<PromptExecutionEvent>(started, Completed(finishReason = null)), events)
    }

    @Test
    fun `cancelling the collector during the stream is a CancellationException, not a mapped network error`() = runBlocking {
        val caught = AtomicReference<Throwable?>()
        val job = launch(Dispatchers.Default) {
            try {
                silentFlow(idleTimeout = 30.seconds).toList()
            } catch (t: Throwable) {
                caught.set(t)
                throw t
            }
        }
        delay(300)
        job.cancelAndJoin()
        assertIs<CancellationException>(assertNotNull(caught.get()))
    }
}
