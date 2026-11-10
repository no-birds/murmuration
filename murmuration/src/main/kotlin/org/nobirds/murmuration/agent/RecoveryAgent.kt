package org.nobirds.murmuration.agent

import org.nobirds.murmuration.internal.logger
import org.nobirds.murmuration.prompt.PromptExecutionException
import kotlin.reflect.KClass

/**
 * Composition agent: primary attempt → match a specific exception type → recovery agent
 * (transforms the input) → retry with the new input. Up to [maxRecoveries] transformations.
 *
 * Use case: image prompt is rejected with PolicyViolationException(PROHIBITED) → a text
 * agent rewrites the character description without gore markers → image generator retries
 * with the cleaned input.
 *
 * If the exception doesn't match [recoverableType] or the recovery limit is exhausted, the
 * original exception propagates. Exceptions thrown by the recovery agent itself propagate
 * as-is — wrapping retry/recovery is the caller's responsibility.
 *
 * Not the same as [RetryAgent]: `RetryAgent` retries the same input on the same error.
 * `RecoveryAgent` performs one transformation step on a specific error type and retries with
 * the new input. Compose freely: `agent.withRecovery<E>(rewriter).withRetry()` recovers via
 * rewrite first, then falls back to plain retry for everything else.
 *
 * For prompt-related failures where the recovery agent needs access to the exception itself
 * (category, finishReason, the failing prompt), use [PromptRecoveryAgent] instead.
 */
class RecoveryAgent<In, Out, E : PromptExecutionException>(
    private val primary: Agent<In, Out>,
    private val recovery: Agent<In, In>,
    private val recoverableType: KClass<E>,
    private val maxRecoveries: Int = 1,
) : WrapperAgent<In, Out> {

    override val delegate: Agent<*, *> get() = primary
    override val name: String get() = primary.name

    private val logger = logger()

    override suspend fun ExecutionContext.execute(input: In): Out {
        reportWrapperLayer("Recovery", primary.name)
        var currentInput = input
        var recoveries = 0
        while (true) {
            try {
                return with(primary) { execute(currentInput) }
            } catch (e: Exception) {
                if (recoveries >= maxRecoveries || !recoverableType.isInstance(e)) throw e
                recoveries++
                logger.info(
                    "Agent {} caught {}, recovering input (attempt {}/{})",
                    primary.name, e::class.simpleName, recoveries, maxRecoveries,
                )
                // Recovery is a separate worker run — fire its START/END via the extension.
                currentInput = execute(recovery, currentInput)
            }
        }
    }
}

/**
 * Convenience wrapper: `agent.withRecovery<PolicyViolationException>(rewriter, maxRecoveries=1)`.
 */
inline fun <In, Out, reified E : PromptExecutionException> Agent<In, Out>.withRecovery(
    recovery: Agent<In, In>,
    maxRecoveries: Int = 1,
): Agent<In, Out> = RecoveryAgent(this, recovery, E::class, maxRecoveries)
