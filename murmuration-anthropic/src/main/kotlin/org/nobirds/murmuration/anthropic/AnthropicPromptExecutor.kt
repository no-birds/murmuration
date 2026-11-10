package org.nobirds.murmuration.anthropic

import com.anthropic.client.AnthropicClient
import com.anthropic.models.messages.MessageCreateParams
import com.anthropic.models.messages.OutputConfig
import com.anthropic.models.messages.ToolUnion
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.emitAll
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.transform
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.InvalidRequestException
import org.nobirds.murmuration.prompt.MessagePart
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptExecutionEvent
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Completed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.Failed
import org.nobirds.murmuration.prompt.PromptExecutionEvent.PartReceived
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.ThinkingLevel
import org.nobirds.murmuration.prompt.ToolsPromptSettings
import org.nobirds.murmuration.prompt.TypedPromptSettings
import org.nobirds.murmuration.prompt.finishReasonOrNull
import org.nobirds.murmuration.prompt.toMessage
import org.nobirds.murmuration.schema.toFunctionSchema
import org.slf4j.LoggerFactory
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

private val logger = LoggerFactory.getLogger(AnthropicPromptExecutor::class.java)

/**
 * [PromptExecutor] for the Claude models of [AnthropicLlms]. Streaming only: a long thinking round-trip would hit
 * the SDK's non-streaming guard, and the stream gives idle/overall timeouts for free. The idle timeout applies only
 * while the thought is streamed (`includeThoughts`); a hidden thought is a silent stream, bounded by the overall budget.
 *
 * One request per [execute], assembled from the ready parts:
 * - `model`: [Llm.model]; `max_tokens`: the prompt's `maxOutputTokens` or the model limit ([AnthropicModelSpec]);
 * - `system` + `messages`: [toAnthropicMessages]; `tools`: [toAnthropicTool] of the prompt's instruments
 *   (`tool_choice` is never sent = `auto`);
 * - structured output (`TypedPromptSettings`): `output_config.format` ([toAnthropicOutputConfig]);
 * - thinking: [AnthropicThinking.forLevel] of the prompt's `thinkingLevel` (none = `MINIMAL`; "off" does not exist; the level is not sent as such, only its form is);
 *   a pure function of the level, so it is the same on every round-trip of a run and never breaks the cache prefix.
 *   The manual `thinkingBudget` is ignored (with a debug log);
 * - `cache_control`: [withCachePoints] from the cache strategies of the prompt parts.
 *
 * Events come from [anthropicEventFlow]; the finished response is judged by [judgeStopReason] (refusal, `max_tokens`,
 * empty answer). Every failure is mapped by [toPromptExecutionException], emitted as [Failed] and rethrown.
 * Usage is emitted before the verdict, so a failed round-trip is still accounted.
 */
class AnthropicPromptExecutor(
    private val client: AnthropicClient,
    private val idleTimeout: Duration = 120.seconds,
    private val overallTimeout: Duration = 600.seconds,
) : PromptExecutor {

    override fun supports(llm: Llm): Boolean = llm.provider == AnthropicLlms.PROVIDER

    override fun execute(prompt: Prompt): Flow<PromptExecutionEvent> = flow {
        var producedContent = false
        prompt.settings?.thinkingBudget?.let {
            logger.debug("thinkingBudget={} ignored for {}: Claude takes the depth from thinkingLevel", it, prompt.llm.model)
        }
        val request = prompt.toRequest()
        val events = anthropicEventFlow(
            openStream = { client.messages().createStreaming(request) },
            requestMessages = prompt.content.map { it.toMessage() },
            requestedModelName = prompt.llm.model,
            // A hidden thought silences the stream legally (the API sends nothing while it thinks): only the overall budget
            // applies then. A visible one streams `thinking_delta`, so silence is a real stall.
            idleTimeout = if (prompt.anthropicThinking().thoughtsVisible) prompt.settings?.streamIdleTimeout ?: idleTimeout else null,
            overallTimeout = overallTimeout,
        )
        emitAll(
            events.transform { event ->
                if (event is PartReceived && event.part.isContent()) producedContent = true
                if (event is Completed) judgeStopReason(prompt, event.finishReason, producedContent)?.let { throw it }
                emit(event)
            }
        )
    }.catch { error ->
        if (error is CancellationException) throw error
        val mapped = error.toPromptExecutionException(prompt)
        emit(Failed(mapped, finishReason = mapped.finishReasonOrNull()))
        throw mapped
    }
}

/**
 * The thinking form of this prompt: depth from `thinkingLevel`, thought streamed (`display = summarized`) when
 * `includeThoughts` is on (the core default when there are no settings). A pure function of the settings.
 */
internal fun Prompt.anthropicThinking(): AnthropicThinking = AnthropicThinking.forLevel(
    spec = AnthropicLlms.findSpec(llm.provider, llm.model),
    level = anthropicThinkingLevel(),
    includeThoughts = settings?.includeThoughts ?: true,
)

/** The tier this prompt goes out with: the asked one, `MINIMAL` when none (Anthropic has no "off"). Also what the RT records. */
internal fun Prompt.anthropicThinkingLevel(): ThinkingLevel = settings?.thinkingLevel ?: ThinkingLevel.MINIMAL

private fun MessagePart.isContent(): Boolean = this !is MessagePart.Text || text.isNotEmpty()

/** Assembles the whole request; schema and history violations are [InvalidRequestException] before the network. */
internal fun Prompt.toRequest(): MessageCreateParams {
    val spec = AnthropicLlms.findSpec(llm.provider, llm.model)
    val settings = settings
    val tools = (settings as? ToolsPromptSettings)?.tools.orEmpty()
        .flatMap { it.instruments }
        .map { it.toFunctionSchema().toAnthropicTool(this) }
    val cached = toAnthropicMessages().withCachePoints(tools)

    val thinking = anthropicThinking()
    val structured = (settings as? TypedPromptSettings<*>)?.toAnthropicOutputConfig(this)
    val effort = thinking.effort

    val builder = MessageCreateParams.builder()
        .model(llm.model)
        .maxTokens((settings?.maxOutputTokens ?: spec.maxOutputTokens).toLong())
        .thinking(thinking.thinkingParam())
    if (structured != null || effort != null) {
        val output: OutputConfig.Builder = structured?.toBuilder() ?: OutputConfig.builder()
        effort?.let { output.effort(it) }
        builder.outputConfig(output.build())
    }
    if (cached.system.isNotEmpty()) builder.system(MessageCreateParams.System.ofTextBlockParams(cached.system))
    if (cached.tools.isNotEmpty()) builder.tools(cached.tools.map { ToolUnion.ofTool(it) })
    return builder.messages(cached.messages).build()
}
