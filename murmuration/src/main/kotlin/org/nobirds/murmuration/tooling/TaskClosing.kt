package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.prompt.*
import org.nobirds.murmuration.llm.*

import kotlinx.serialization.json.JsonElement
import kotlin.reflect.KType
import kotlin.reflect.typeOf

/**
 * Marks a `@Instrument` function that should NOT have an auto-injected `closesTaskId`
 * parameter. Used for instruments like `transitionTo` or `completeTask` itself, which are
 * not domain actions and therefore must not close a task implicitly.
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
annotation class CantCloseTask

/**
 * Contract for a builder/tool that owns a set of tasks closable from an instrument call.
 *
 * When the owner implements this interface, [TaskClosingDecorator] adds a `closesTaskId`
 * parameter to every `@Instrument` function not marked with [CantCloseTask].
 */
interface TaskClosing {
    /**
     * Marks the task/issue with the given ID as closed and returns a string response for
     * the LLM. Also reports whether any unclosed tasks remain.
     */
    suspend fun closeTaskFromTool(id: String): TaskCloseResult

    /**
     * Called when a call carrying `closesTaskId = id` FAILED (the [ErrorMatcher] classified its
     * result as an error, see [TaskClosingInstrumentDeclaration.invoke]) — the task was therefore
     * NOT closed by [closeTaskFromTool]. Default no-op; a batch-guard owner (like
     * `TaskAwareBuilder`) overrides this to remember "id failed in THIS batch" so a sibling
     * `completeTask(id)` call in the same batch can be rejected instead of silently closing a
     * task whose actual mutation never happened.
     */
    suspend fun notifyTaskCloseFailed(id: String) {}
}

data class TaskCloseResult(
    /** Message for the LLM (matches what an inline `completeTask` instrument would return). */
    val message: String,
    /**
     * True when no tasks remain after this close. The phase switch itself is applied at the
     * END of the round-trip (transitions apply between round-trips, see
     * [Stateful.onAfterToolCalls]) — this flag only signals that the closing call emptied the
     * task list, so the wrapper can hint the model that the phase will advance.
     */
    val allTasksClosed: Boolean,
)

/**
 * Decorator that adds an optional `closesTaskId` parameter and the task-closing logic for
 * owners implementing [TaskClosing]. Activates only when the owner is `TaskClosing` and
 * the function is not marked with [CantCloseTask]. Sits outermost among the defaults so
 * that the [ErrorMatcher] inspects the post-validator result — a `Reject` message from a
 * `ParameterValidator` correctly does NOT close the task.
 *
 * The "error vs ok" classification comes from [ErrorMatcher.resolveDefault] — the single source
 * shared by every classification path of the framework.
 */
class TaskClosingDecorator : InstrumentDecorator {
    override val order: Int = 30

    override fun appliesTo(instrument: InstrumentDeclaration<*>, owner: Any): Boolean =
        owner is TaskClosing && instrument.annotations.none { it is CantCloseTask }

    override fun decorate(instrument: InstrumentDeclaration<*>, owner: Any): InstrumentDeclaration<*> =
        TaskClosingInstrumentDeclaration<Any?>(instrument, owner as TaskClosing)

    companion object {
        /** Default instance for [DecoratorRegistry.Default]. */
        val Default: TaskClosingDecorator = TaskClosingDecorator()
    }
}

/**
 * Wrapper that supports `closesTaskId`. Re-declares parameters (adds one optional), and
 * after the inner invoke decides whether to call [TaskClosing.closeTaskFromTool]. The
 * "error vs ok" classification is delegated to [ErrorMatcher.resolveDefault].
 */
private class TaskClosingInstrumentDeclaration<R>(
    private val delegate: InstrumentDeclaration<*>,
    private val closer: TaskClosing,
) : InstrumentDeclaration<R> {
    override val obj: Any = delegate.obj
    override val name: String = delegate.name
    override val description: String = delegate.description
    override val annotations: List<Annotation> = delegate.annotations
    override val returnType: KType = delegate.returnType

    override val parameters: List<InstrumentParameter> = delegate.parameters + InstrumentParameter(
        name = CLOSES_TASK_ID_PARAM,
        type = typeOf<String?>(),
        description = "Optional. ID of the task or issue that is FULLY resolved by THIS call — pass it OPTIMISTICALLY when you are confident in the arguments. If the call returns an error, the wrapper itself will NOT close the task. Don't use it for partial progress; for multi-tool tasks, close via completeTask after the final step.",
        isOptional = true,
    )

    override fun prepareWireArguments(args: Map<String, JsonElement>): Map<String, Any?> =
        delegate.prepareWireArguments(args)

    override suspend fun invoke(args: Map<String, Any?>): R {
        val closesTaskId = (args[CLOSES_TASK_ID_PARAM] as? String)?.takeIf { it.isNotBlank() }
        val cleanedArgs = if (args.containsKey(CLOSES_TASK_ID_PARAM)) args - CLOSES_TASK_ID_PARAM else args
        @Suppress("UNCHECKED_CAST")
        val result = delegate.invoke(cleanedArgs) as R
        if (closesTaskId == null) return result

        if (ErrorMatcher.resolveDefault().isError(result)) {
            closer.notifyTaskCloseFailed(closesTaskId)
            return result
        }

        val closeResult = closer.closeTaskFromTool(closesTaskId)
        if (returnType.classifier != String::class) return result

        val combined = buildString {
            append(result?.toString().orEmpty())
            append("\n[").append(closeResult.message).append("]")
            if (closeResult.allTasksClosed) {
                append("\n[All tasks and issues are resolved — the phase will switch at the end of this response.]")
            }
        }
        @Suppress("UNCHECKED_CAST")
        return combined as R
    }

    companion object {
        const val CLOSES_TASK_ID_PARAM = "closesTaskId"
    }
}
