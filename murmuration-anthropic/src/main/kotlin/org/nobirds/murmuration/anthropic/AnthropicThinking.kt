package org.nobirds.murmuration.anthropic

import com.anthropic.models.messages.OutputConfig
import com.anthropic.models.messages.ThinkingConfigAdaptive
import com.anthropic.models.messages.ThinkingConfigBetweenTools
import com.anthropic.models.messages.ThinkingConfigParam
import org.nobirds.murmuration.prompt.ThinkingLevel

/**
 * The thinking form sent to the API for one request. Claude 5.5 rejects the manual form
 * (`enabled` + budget) with 400, so only these two forms exist; "thinking off" is not
 * expressible (product decision): the least a model can do is its [AnthropicModelSpec.minimalThinking].
 */
sealed interface AnthropicThinking {

    /** Wire `thinking` parameter of this form. */
    fun thinkingParam(): ThinkingConfigParam

    /** Wire `output_config.effort`; `null` when the form carries no depth. */
    val effort: OutputConfig.Effort?

    /**
     * Whether the request makes the API stream the thought (`thinking.display = "summarized"`). By default Claude hides the
     * thought and the SSE stream goes fully silent while it thinks — so a hidden thought is a legal long pause, a visible one
     * is a steady flow of `thinking_delta` events.
     */
    val thoughtsVisible: Boolean

    /** This form with the thought shown or hidden; a form that cannot show it ([BetweenTools]) stays as it is. */
    fun withThoughts(visible: Boolean): AnthropicThinking

    /** `between_tools`: no thinking before the answer, short updates between tool calls. Sonnet/Opus minimum. */
    data object BetweenTools : AnthropicThinking {
        override fun thinkingParam(): ThinkingConfigParam =
            ThinkingConfigParam.ofBetweenTools(ThinkingConfigBetweenTools.builder().build())

        override val effort: OutputConfig.Effort? = null

        /** The form has no `display` field: its thought is never streamed. */
        override val thoughtsVisible: Boolean = false

        override fun withThoughts(visible: Boolean): AnthropicThinking = this
    }

    /** `adaptive` with a depth: the model decides whether and how much to think, bounded by [effort]. */
    data class Adaptive(override val effort: OutputConfig.Effort, override val thoughtsVisible: Boolean = false) : AnthropicThinking {
        override fun thinkingParam(): ThinkingConfigParam {
            val builder = ThinkingConfigAdaptive.builder()
            if (thoughtsVisible) builder.display(ThinkingConfigAdaptive.Display.SUMMARIZED)
            return ThinkingConfigParam.ofAdaptive(builder.build())
        }

        override fun withThoughts(visible: Boolean): AnthropicThinking = copy(thoughtsVisible = visible)
    }

    companion object {
        /**
         * Maps a core [level] onto a form for [spec]. `MINIMAL` is the model's minimal form; the
         * other levels are `adaptive` with effort LOW / MEDIUM / HIGH. `XHIGH` and `MAX` are not
         * reachable from the core levels on purpose (cost). The manual `thinkingBudget` is not
         * consulted anywhere: the form has no place for it. [includeThoughts] asks the API to stream the thought
         * (see [thoughtsVisible]).
         */
        fun forLevel(spec: AnthropicModelSpec, level: ThinkingLevel, includeThoughts: Boolean = false): AnthropicThinking =
            forLevelHidden(spec, level).withThoughts(includeThoughts)

        private fun forLevelHidden(spec: AnthropicModelSpec, level: ThinkingLevel): AnthropicThinking = when (level) {
            ThinkingLevel.MINIMAL -> spec.minimalThinking
            ThinkingLevel.LOW -> Adaptive(OutputConfig.Effort.LOW)
            ThinkingLevel.MEDIUM -> Adaptive(OutputConfig.Effort.MEDIUM)
            ThinkingLevel.HIGH -> Adaptive(OutputConfig.Effort.HIGH)
        }
    }
}
