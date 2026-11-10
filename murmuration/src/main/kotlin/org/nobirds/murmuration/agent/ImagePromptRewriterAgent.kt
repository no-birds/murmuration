package org.nobirds.murmuration.agent

import org.nobirds.murmuration.internal.logger
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.PolicyViolationException
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptExecutionException
import org.nobirds.murmuration.prompt.PromptExecutionStatistics
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.errorCode
import org.nobirds.murmuration.prompt.finishReasonOrNull
import org.nobirds.murmuration.prompt.TextPromptExecutionResult
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.aggregate
import org.nobirds.murmuration.prompt.buildPrompt
import org.nobirds.murmuration.prompt.system
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import kotlin.time.TimeSource

/**
 * `Agent<PolicyViolationException, Prompt>` that rewrites a rejected image prompt with a
 * text model. Used as the `recovery` in [PromptRecoveryAgent] around [ImageAgent] to
 * recover from [PolicyViolationException] thrown by Imagen filters.
 *
 * The input is the exception itself: the rewriter reads both the failing [Prompt]
 * ([PolicyViolationException.prompt]) and the rejection category
 * ([PolicyViolationException.category]) plus the raw provider `finishReason`. This lets it
 * pick a category-specific instruction (via [instructionFor]) — PROHIBITED typically needs
 * aggressive gore-marker stripping, SAFETY needs general tone softening, and so on.
 *
 * A single instance can be shared across all image agents in a project (NPC portraits,
 * tokens, maps): the rewriter knows nothing about domain types — it works on prompt text
 * and the rejection category.
 */
class ImagePromptRewriterAgent(
    private val textExecutor: PromptExecutor,
    private val textLlm: Llm,
    private val instructionFor: (PolicyViolationException) -> String = ::defaultInstruction,
    name: String = "ImagePromptRewriter",
) : Agent<PolicyViolationException, Prompt> {

    override val name: String = generateAgentName(name, "PromptRewriter")
    private val logger = logger()

    override suspend fun ExecutionContext.execute(input: PolicyViolationException): Prompt {
        val displayName = getDisplayName(name)
        val originalPrompt = input.prompt
        logger.debug(
            "Agent {} started for category={}, finishReason={}",
            displayName, input.category, input.finishReason,
        )

        val originalText = extractUserText(originalPrompt)
        if (originalText.isBlank()) {
            error("ImagePromptRewriter: input prompt has no USER text to rewrite")
        }

        val instruction = instructionFor(input)
        val rewritePrompt = buildPrompt {
            llm = textLlm
            system { text(instruction) }
            user { text("ORIGINAL IMAGE PROMPT:\n$originalText") }
        }

        val startTime = TimeSource.Monotonic.markNow()
        try {
            checkCallLimit()
            val result = textExecutor.execute(rewritePrompt).aggregate() as? TextPromptExecutionResult
                ?: throw PromptExecutorException(
                    "ImagePromptRewriter expected TextPromptExecutionResult, got different type"
                )
            val duration = startTime.elapsedNow()

            reportAgentExecution(PromptExecutionEntry(displayName, duration, result.statistics))

            val rewrittenText = result.text.trim()
            if (rewrittenText.isBlank()) {
                error("ImagePromptRewriter: text model returned empty rewrite")
            }

            val newPrompt = buildPrompt {
                llm = originalPrompt.llm
                settings = originalPrompt.settings
                user { text(rewrittenText) }
            }

            logger.debug(
                "Agent {} finished in {}, text {} → {} chars",
                displayName, duration, originalText.length, rewrittenText.length,
            )
            return newPrompt
        } catch (e: Exception) {
            val duration = startTime.elapsedNow()
            val typed = e as? PromptExecutionException
            // Partial statistics already assembled before the failure (tokens, responseMessage,
            // model name) so the FAILURE row reflects the round-trip's real partial content
            // instead of losing tokens entirely, as this call used to (sixth of six
            // failure-statistics call sites the pattern was inconsistent at).
            val partial = typed?.partialStats
            val stats = (partial ?: PromptExecutionStatistics()).copy(
                success = false,
                errorMessage = e.message,
                attempts = 1,
                errorCode = typed?.errorCode(),
                // `?:` — `finishReasonOrNull()`/`streamDiagnostics`
                // are non-null only for a `PolicyViolationException`; on any other exception type
                // this would otherwise clobber a value `partial` already carried.
                finishReason = typed?.finishReasonOrNull() ?: partial?.finishReason,
                streamDiagnostics = typed?.streamDiagnostics ?: partial?.streamDiagnostics,
            )
            reportAgentExecution(PromptExecutionEntry(displayName, duration, stats))
            logger.debug("Agent {} failed in {}. Error: {}", displayName, duration, e.message)
            throw e
        }
    }

    private fun extractUserText(prompt: Prompt): String =
        prompt.content
            .filter { it.type == PromptContentType.USER }
            .flatMap { it.parts }
            .filterIsInstance<TextPromptPart>()
            .joinToString("\n") { it.text }
}

/**
 * Default instruction template for the rewriter. Substitutes the rejection category and
 * provider `finishReason` into a generic prompt — the model sees what was rejected and
 * adapts.
 *
 * Customize via the `instructionFor` constructor parameter: pass a domain-specific text
 * (e.g. medical illustration permits more explicit content), or branch per category.
 */
fun defaultInstruction(e: PolicyViolationException): String = """
You are an editor for AI image-generation prompts. The given prompt was rejected by content moderation.

Rejection details:
- category: ${e.category.name}
- provider reason: ${e.finishReason ?: "unspecified"}

Rephrase the prompt so it passes moderation while preserving:
- the subject's identity (race, class, role),
- the composition and pose,
- equipment, clothing, materials, and lighting,
- the overall atmosphere and tone.

Adjust the language to remove markers that triggered the rejection, but do not invent new content or change the meaning. Output ONLY the rewritten prompt as a single plain-text message — no preamble, no explanations, no markdown.
""".trimIndent()
