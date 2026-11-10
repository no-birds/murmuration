package org.nobirds.murmuration.tooling

import ch.qos.logback.classic.Logger
import ch.qos.logback.classic.spi.ILoggingEvent
import ch.qos.logback.core.read.ListAppender
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.agent.RetryAgent
import org.nobirds.murmuration.agent.agent
import org.nobirds.murmuration.agent.execute
import org.nobirds.murmuration.agent.executePipeline
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Failed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PromptStarted
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.QuotaExceededException
import org.nobirds.murmuration.prompt.QuotaKind
import org.nobirds.murmuration.prompt.TransientException
import org.nobirds.murmuration.prompt.TransientReason
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.retry.DefaultRetryStrategy
import org.nobirds.murmuration.retry.RetryPolicy
import org.slf4j.LoggerFactory
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds

/**
 * the retry log line reports the attempt INSIDE the budget of its error class
 * ("2/2"), never the overall counter ("4/2"). Budgets are per class (quota / other), so with
 * `maxRetries = 2` two quota failures followed by two timeouts are all retried and the lines read
 * 1/2, 2/2, 1/2, 2/2.
 *
 * RED on HEAD: `RetryAgent` and `ToolCallingAgent` log the overall `session.attempt` (3/2, 4/2).
 */
class RetryLogAttemptTest {

    private val policy = RetryPolicy(
        strategy = DefaultRetryStrategy,
        maxRetries = 2,
        initialBackoff = 1.milliseconds,
        maxBackoff = 2.milliseconds,
        quotaBackoffs = listOf(1.milliseconds),
    )

    private val fakeLlm = Llm(provider = "test", family = "test", name = "test", version = "1", stable = true, model = "test-model")

    class NoopTool {
        @Instrument("Always succeeds.")
        fun ping(): String = "pong"
    }

    private val appender = ListAppender<ILoggingEvent>()
    private val root = LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME) as Logger

    @BeforeEach
    fun attach() {
        appender.start()
        root.addAppender(appender)
    }

    @AfterEach
    fun detach() {
        root.detachAppender(appender)
        appender.stop()
    }

    private fun quota() = QuotaExceededException(QuotaKind.RATE_LIMIT, retryAfter = null, message = "429 RESOURCE_EXHAUSTED", cause = null)

    private fun timeout() = TransientException(TransientReason.NETWORK_TIMEOUT, "ttfb", null)

    /** Failure sequence: quota, quota, timeout, timeout, then success. */
    private fun failureFor(call: Int): Throwable? = when (call) {
        1, 2 -> quota()
        3, 4 -> timeout()
        else -> null
    }

    private val attemptLine = Regex("(?:attempt|retry) (\\d+)/(\\d+)")

    private fun loggedAttempts(): List<Pair<Int, Int>> = appender.list
        .map { it.formattedMessage }
        .mapNotNull { attemptLine.find(it) }
        .map { it.groupValues[1].toInt() to it.groupValues[2].toInt() }

    private fun assertWithinBudget(attempts: List<Pair<Int, Int>>) {
        assertEquals(4, attempts.size, "four retries expected; log=$attempts")
        assertTrue(attempts.all { (k, max) -> k <= max }, "attempt must be counted inside its class budget (k<=max): $attempts")
        assertEquals(listOf(1, 2, 1, 2), attempts.map { it.first })
    }

    @Test
    fun `RetryAgent logs the attempt inside the budget of its error class`() = runBlocking {
        val calls = AtomicInteger()
        val flaky = agent<Unit, String>("Flaky") {
            failureFor(calls.incrementAndGet())?.let { throw it }
            "ok"
        }
        executePipeline { execute(RetryAgent(flaky, policy), Unit) }

        assertWithinBudget(loggedAttempts())
    }

    @Test
    fun `ToolCallingAgent round-trip retry logs the attempt inside the budget of its error class`() = runBlocking {
        val calls = AtomicInteger()
        val executor = PromptExecutor {
            flow {
                val error = failureFor(calls.incrementAndGet())
                emit(PromptStarted(modelName = "test-model", requestedModelName = "test-model", requestMessages = emptyList()))
                if (error != null) {
                    emit(Failed(error))
                } else {
                    emit(PartReceived(MessagePart.Text("done")))
                    emit(Completed(finishReason = "STOP"))
                }
            }
        }
        executePipeline {
            execute(
                toolCallingAgent<Unit, NoopTool>(
                    llm = fakeLlm, executor = executor, name = "RetryLogAgent", maxTurns = 4, roundTripRetry = policy,
                ) { _, _, _ -> user { text("test prompt") } },
                ToolCallingInput(Unit, NoopTool()),
            )
        }

        assertWithinBudget(loggedAttempts())
    }
}
