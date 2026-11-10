package org.nobirds.murmuration.agent

import org.nobirds.murmuration.internal.logger
import org.nobirds.murmuration.prompt.Prompt
import org.nobirds.murmuration.prompt.PromptRelatedException
import kotlin.reflect.KClass

/**
 * Specialized [RecoveryAgent] for prompt-related failures: the recovery agent receives the
 * **exception** ([E]) instead of the previous input. Through the exception it reads both
 * the failing prompt ([PromptRelatedException.prompt]) and the formal cause (e.g. category /
 * finishReason for [org.nobirds.murmuration.prompt.PolicyViolationException]) — and returns
 * a new [Prompt] for the next primary attempt.
 *
 * Why a separate class: the generic [RecoveryAgent]'s recovery is `Agent<In, In>` — it knows
 * nothing about the exception, only the previous input. For an image-rewriter, access to the
 * rejection category is critical (different categories may want different rewrite
 * instructions), so the recovery agent has to receive the typed exception.
 *
 * Not interchangeable with [RecoveryAgent]: that one covers cases where there is no
 * "exception ↔ prompt" link. Here the link is explicit: primary works on [Prompt], recovery
 * receives the exception → returns a new [Prompt].
 */
class PromptRecoveryAgent<Out, E : PromptRelatedException>(
    private val primary: Agent<Prompt, Out>,
    private val recovery: Agent<E, Prompt>,
    private val recoverableType: KClass<E>,
    private val maxRecoveries: Int = 1,
) : WrapperAgent<Prompt, Out> {

    // Wrapper is transparent to the event stream. Outer extension already fired START
    // with primary.name; we only invoke primary via raw `with(...)`, never the extension.
    override val delegate: Agent<*, *> get() = primary
    override val name: String get() = primary.name

    private val logger = logger()

    override suspend fun ExecutionContext.execute(input: Prompt): Out {
        reportWrapperLayer("PromptRecovery", primary.name)
        var currentInput = input
        var recoveries = 0
        while (true) {
            try {
                // Raw call — deliberate, no double START/END.
                return with(primary) { execute(currentInput) }
            } catch (e: Exception) {
                if (recoveries >= maxRecoveries || !recoverableType.isInstance(e)) throw e
                @Suppress("UNCHECKED_CAST")
                val typed = e as E
                recoveries++
                logger.info(
                    "Agent {} caught {}, rewriting prompt (attempt {}/{})",
                    primary.name, e::class.simpleName, recoveries, maxRecoveries,
                )
                // Recovery is a separate piece of work (rewriter LLM call) — fire its own
                // START/END via the instrumented extension. UI will count it as +1 worker.
                currentInput = execute(recovery, typed)
            }
        }
    }
}

/**
 * Convenience wrapper for prompt-related recovery:
 * `imagePromptAgent.withPromptRecovery<PolicyViolationException>(rewriter)`. The recovery
 * agent receives the exception and returns a new [Prompt].
 */
inline fun <Out, reified E : PromptRelatedException> Agent<Prompt, Out>.withPromptRecovery(
    recovery: Agent<E, Prompt>,
    maxRecoveries: Int = 1,
): Agent<Prompt, Out> = PromptRecoveryAgent(this, recovery, E::class, maxRecoveries)
