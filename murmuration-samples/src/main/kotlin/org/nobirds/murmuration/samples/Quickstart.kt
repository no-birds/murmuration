package org.nobirds.murmuration.samples

// region quickstart
import org.nobirds.murmuration.agent.Agent
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.tooling.Instrument
import org.nobirds.murmuration.tooling.InstrumentParam
import org.nobirds.murmuration.tooling.toolAgent

/** A tool is a plain class: every `@Instrument` method becomes a function the model can call. */
class Calculator {
    var lastSum: Int? = null

    @Instrument("Add two integers and return the sum.")
    fun add(
        @InstrumentParam("First addend") a: Int,
        @InstrumentParam("Second addend") b: Int,
    ): Int = (a + b).also { lastSum = it }
}

/**
 * Builds an agent that lets the model use [Calculator]. The provider is just a [PromptExecutor]:
 * pass `GeminiPromptExecutor`, an OpenAI or an Anthropic executor, or a fake one in tests.
 */
fun calculatorAgent(llm: Llm, executor: PromptExecutor): Agent<String, Int?> =
    toolAgent(
        llm = llm,
        executor = executor,
        tool = Calculator(),
        transform = { _, calculator -> calculator.lastSum },
    ) { _, question, _ ->
        user { text(question) }
    }
// endregion
