package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.prompt.*
import org.nobirds.murmuration.llm.*

import kotlinx.serialization.json.JsonElement
import kotlin.reflect.KType

/**
 * SPI for instrument decorators: the factory builds a raw [InstrumentDeclaration] from
 * reflection and runs it through a chain of decorators (ordered by [order] ascending —
 * smaller values are closer to the raw invocation).
 *
 * Decorators are source-agnostic: they only see the generic [InstrumentDeclaration], its
 * [annotations], and per-parameter annotations through [InstrumentParameter.annotations].
 * That makes it possible to add arbitrary aspects (validation, limits, metrics,
 * post-processing) without caring how the instrument was produced (KFunction reflection,
 * hand-written declaration, etc.).
 */
interface InstrumentDecorator {
    val order: Int
    fun appliesTo(instrument: InstrumentDeclaration<*>, owner: Any): Boolean
    fun decorate(instrument: InstrumentDeclaration<*>, owner: Any): InstrumentDeclaration<*>
}

/**
 * Chain of [InstrumentDecorator]s applied by [InstrumentFactory] to every instrument
 * assembled from a `@Instrument` function.
 */
class DecoratorRegistry(decorators: List<InstrumentDecorator>) {
    private val sorted = decorators.sortedBy { it.order }

    fun apply(instrument: InstrumentDeclaration<*>, owner: Any): InstrumentDeclaration<*> {
        var current: InstrumentDeclaration<*> = instrument
        for (decorator in sorted) {
            if (decorator.appliesTo(current, owner)) {
                current = decorator.decorate(current, owner)
            }
        }
        return current
    }

    companion object {
        /**
         * Default chain: the universal SPI decorators (parameter validation and
         * cross-parameter validation) plus the bundled [TaskClosingDecorator]. Application
         * code that needs a different mix can construct its own [DecoratorRegistry].
         */
        val Default: DecoratorRegistry = DecoratorRegistry(listOf(
            ParameterTransformingDecorator(TransformerRegistry.Default),
            ParameterValidatingDecorator(ValidatorRegistry.Default),
            CallValidatingDecorator,
            TaskClosingDecorator.Default,
        ))
    }
}

/**
 * Decorator that applies [ParameterTransformer]s to parameters whose annotations are
 * registered in [registry]. Runs **before** [ParameterValidatingDecorator] (order 5 vs 10)
 * so validators see the post-transform value (e.g. `@NotBlank` + `@Trim` correctly rejects
 * `"   "` because the trim runs first and produces `""`).
 *
 * Source-agnostic — reads parameter annotations through [InstrumentParameter.annotations].
 */
class ParameterTransformingDecorator(
    private val registry: TransformerRegistry,
) : InstrumentDecorator {
    override val order: Int = 5

    override fun appliesTo(instrument: InstrumentDeclaration<*>, owner: Any): Boolean =
        collectTransformers(instrument).isNotEmpty()

    override fun decorate(instrument: InstrumentDeclaration<*>, owner: Any): InstrumentDeclaration<*> =
        TransformingInstrumentDeclaration<Any?>(instrument, collectTransformers(instrument))

    private fun collectTransformers(instrument: InstrumentDeclaration<*>): Map<String, List<ParameterTransformer>> {
        val result = mutableMapOf<String, MutableList<ParameterTransformer>>()
        for (parameter in instrument.parameters) {
            for (annotation in parameter.annotations) {
                val transformers = registry.resolve(annotation)
                if (transformers.isNotEmpty()) {
                    result.getOrPut(parameter.name) { mutableListOf() } += transformers
                }
            }
        }
        return result
    }
}

private class TransformingInstrumentDeclaration<R>(
    private val delegate: InstrumentDeclaration<*>,
    private val transformers: Map<String, List<ParameterTransformer>>,
) : InstrumentDeclaration<R> {
    override val obj: Any = delegate.obj
    override val name: String = delegate.name
    override val description: String = delegate.description
    override val parameters: List<InstrumentParameter> = delegate.parameters
    override val annotations: List<Annotation> = delegate.annotations
    override val returnType: KType = delegate.returnType

    override fun prepareWireArguments(args: Map<String, JsonElement>): Map<String, Any?> =
        delegate.prepareWireArguments(args)

    override suspend fun invoke(args: Map<String, Any?>): R {
        val rewritten = args.toMutableMap()
        for ((paramName, paramTransformers) in transformers) {
            if (paramName !in rewritten) continue
            var value: Any? = rewritten[paramName]
            for (transformer in paramTransformers) {
                value = transformer.transform(value)
            }
            rewritten[paramName] = value
        }
        @Suppress("UNCHECKED_CAST")
        return delegate.invoke(rewritten) as R
    }
}

/**
 * Decorator that applies [ParameterValidator]s to parameters whose annotations are
 * registered in [registry]. Source-agnostic — reads parameter annotations through
 * [InstrumentParameter.annotations].
 */
class ParameterValidatingDecorator(
    private val registry: ValidatorRegistry,
) : InstrumentDecorator {
    override val order: Int = 10

    override fun appliesTo(instrument: InstrumentDeclaration<*>, owner: Any): Boolean =
        collectValidators(instrument).isNotEmpty()

    override fun decorate(instrument: InstrumentDeclaration<*>, owner: Any): InstrumentDeclaration<*> =
        ValidatingInstrumentDeclaration<Any?>(instrument, collectValidators(instrument))

    private fun collectValidators(instrument: InstrumentDeclaration<*>): Map<String, List<ParameterValidator>> {
        val result = mutableMapOf<String, MutableList<ParameterValidator>>()
        for (parameter in instrument.parameters) {
            for (annotation in parameter.annotations) {
                val validators = registry.resolve(annotation)
                if (validators.isNotEmpty()) {
                    result.getOrPut(parameter.name) { mutableListOf() } += validators
                }
            }
        }
        return result
    }
}

private class ValidatingInstrumentDeclaration<R>(
    private val delegate: InstrumentDeclaration<*>,
    private val validators: Map<String, List<ParameterValidator>>,
) : InstrumentDeclaration<R> {
    override val obj: Any = delegate.obj
    override val name: String = delegate.name
    override val description: String = delegate.description
    override val parameters: List<InstrumentParameter> = delegate.parameters
    override val annotations: List<Annotation> = delegate.annotations
    override val returnType: KType = delegate.returnType

    override fun prepareWireArguments(args: Map<String, JsonElement>): Map<String, Any?> =
        delegate.prepareWireArguments(args)

    override suspend fun invoke(args: Map<String, Any?>): R {
        for ((paramName, paramValidators) in validators) {
            val value = args[paramName]
            for (validator in paramValidators) {
                val result = validator.validate(paramName, value)
                if (result is ValidationResult.Reject) {
                    check(returnType.classifier == String::class) {
                        "ParameterValidator on '$paramName' of '${delegate.name}' rejected, but return type is $returnType (must be String to deliver the reject message to LLM)"
                    }
                    @Suppress("UNCHECKED_CAST")
                    return result.message as R
                }
            }
        }
        @Suppress("UNCHECKED_CAST")
        return delegate.invoke(args) as R
    }
}

/**
 * Decorator that applies the [CallValidator]s listed by [CallValidatedBy] on a function.
 * Source-agnostic — reads the annotation through [InstrumentDeclaration.annotations].
 */
object CallValidatingDecorator : InstrumentDecorator {
    override val order: Int = 20

    override fun appliesTo(instrument: InstrumentDeclaration<*>, owner: Any): Boolean {
        val ann = instrument.annotations.filterIsInstance<CallValidatedBy>().firstOrNull() ?: return false
        return ann.validators.isNotEmpty()
    }

    override fun decorate(instrument: InstrumentDeclaration<*>, owner: Any): InstrumentDeclaration<*> {
        val ann = instrument.annotations.filterIsInstance<CallValidatedBy>().first()
        val validators = ann.validators.mapNotNull { it.objectInstance }
        return CallValidatingInstrumentDeclaration<Any?>(instrument, validators)
    }
}

private class CallValidatingInstrumentDeclaration<R>(
    private val delegate: InstrumentDeclaration<*>,
    private val validators: List<CallValidator>,
) : InstrumentDeclaration<R> {
    override val obj: Any = delegate.obj
    override val name: String = delegate.name
    override val description: String = delegate.description
    override val parameters: List<InstrumentParameter> = delegate.parameters
    override val annotations: List<Annotation> = delegate.annotations
    override val returnType: KType = delegate.returnType

    override fun prepareWireArguments(args: Map<String, JsonElement>): Map<String, Any?> =
        delegate.prepareWireArguments(args)

    override suspend fun invoke(args: Map<String, Any?>): R {
        for (validator in validators) {
            val result = validator.validate(this, args)
            if (result is ValidationResult.Reject) {
                check(returnType.classifier == String::class) {
                    "@CallValidatedBy on '${delegate.name}' rejected, but return type is $returnType (must be String to deliver the reject message to LLM)"
                }
                @Suppress("UNCHECKED_CAST")
                return result.message as R
            }
        }
        @Suppress("UNCHECKED_CAST")
        return delegate.invoke(args) as R
    }
}
