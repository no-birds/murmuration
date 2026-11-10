package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.prompt.*
import org.nobirds.murmuration.schema.JsonSchemaBuilder

import kotlin.reflect.KClass

/**
 * Result of setter-time parameter validation or cross-parameter call validation.
 *
 * [Reject.message] is returned to the LLM as a normal successful tool response (typically
 * prefixed with the agreed error marker, see [ErrorMatcher]), so that
 * `TaskClosingDecorator` will not close the task and the LLM gets a clear "call again with
 * a corrected value" signal.
 */
sealed interface ValidationResult {
    data object Ok : ValidationResult
    data class Reject(val message: String) : ValidationResult
}

/**
 * SPI for a single-parameter validator. Registered in [ValidatorRegistry] by annotation
 * type; the runtime wires it in during instrument assembly through [InstrumentDecorator].
 *
 * A validator carries three concerns in one place:
 * 1. [validate] — the runtime check applied before the delegate runs.
 * 2. [applyToSchema] — projection to JSON-Schema so the LLM sees the constraint up-front
 *    (e.g. `minLength`, `pattern`, `minItems`). Default no-op.
 * 3. [describe] — a short human-readable summary of the constraint, mixed into the
 *    `description` field of the generated schema so the LLM has both formal rules and
 *    a natural-language hint. Default empty.
 */
interface ParameterValidator {
    fun validate(name: String, value: Any?): ValidationResult
    fun applyToSchema(builder: JsonSchemaBuilder) {}
    fun describe(): String = ""
}

/**
 * SPI for a cross-parameter call validator. Sees the full argument map — useful for
 * "if type=CREATE then id is required" style checks. Attached via [CallValidatedBy] on a
 * `@Instrument` function.
 */
fun interface CallValidator {
    fun validate(declaration: InstrumentDeclaration<*>, args: Map<String, Any?>): ValidationResult
}

/**
 * Tags a `@Instrument` function with a list of [CallValidator] singleton objects. The
 * runtime invokes them before the delegate; if any returns [ValidationResult.Reject], the
 * message is returned as the call result (the function must return a String-compatible
 * type).
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
annotation class CallValidatedBy(vararg val validators: KClass<out CallValidator>)

/**
 * Registry of parameter-validator factories keyed by annotation type. The annotation
 * instance is passed to the factory, which decides what to extract from it (e.g.
 * [MinLength] forwards `value`, [NotContains] forwards the `forbidden` list).
 *
 * To add your own `@MyValidation(...)` outside the framework:
 * 1. Declare the annotation with `@Retention(RUNTIME)` and `@Target(VALUE_PARAMETER)`.
 * 2. Register a factory: `registry.register<MyValidation> { ann -> MyValidator(ann.foo) }`.
 * 3. Place the annotation on a `@Instrument` parameter — the runtime assembles the
 *    validating wrapper automatically.
 *
 * Composing — annotations marked with [ValidationGroup] are expanded recursively:
 * [resolve] walks meta-annotations on the class itself and collects all transitively
 * registered validators (with cycle protection). Use to build domain shortcuts on top
 * of the universal building blocks ([NotBlank], [Pattern], [NotContains], …).
 */
class ValidatorRegistry {
    @PublishedApi
    internal val factories = mutableMapOf<KClass<out Annotation>, (Annotation) -> ParameterValidator>()

    inline fun <reified A : Annotation> register(noinline factory: (A) -> ParameterValidator) = apply {
        @Suppress("UNCHECKED_CAST")
        factories[A::class] = factory as (Annotation) -> ParameterValidator
    }

    /**
     * Returns a validator for the given annotation if a factory is registered for its
     * type, `null` otherwise. Does NOT expand [ValidationGroup] composites — use
     * [resolve] for the full transitive list.
     */
    fun build(annotation: Annotation): ParameterValidator? =
        factories[annotation.annotationClass]?.invoke(annotation)

    /**
     * Returns the full transitive list of validators for an annotation: the validator
     * registered for the annotation type itself (if any), plus — when the annotation is
     * a [ValidationGroup] composite — validators for each meta-annotation declared on it.
     *
     * Cycle-safe: a composite that (transitively) references itself stops the walk.
     */
    fun resolve(annotation: Annotation): List<ParameterValidator> {
        val out = mutableListOf<ParameterValidator>()
        val visited = mutableSetOf<KClass<out Annotation>>()
        resolveInto(annotation, out, visited)
        return out
    }

    private fun resolveInto(
        annotation: Annotation,
        out: MutableList<ParameterValidator>,
        visited: MutableSet<KClass<out Annotation>>
    ) {
        val type = annotation.annotationClass
        if (!visited.add(type)) return
        // Direct validator (if registered) — most specific is added first.
        factories[type]?.invoke(annotation)?.let(out::add)
        // Composite — walk meta-annotations on the annotation class.
        if (type.annotations.any { it is ValidationGroup }) {
            for (meta in type.annotations) {
                if (meta is ValidationGroup) continue
                resolveInto(meta, out, visited)
            }
        }
    }

    companion object {
        /**
         * Default registry: universal validators ([NotBlank], [NotEmpty], [MinLength],
         * [MaxLength], [Pattern], [NotContains], [MinSize], [MaxSize], [Min], [Max]).
         * Application code registers its own product-specific validators on top.
         */
        val Default: ValidatorRegistry = ValidatorRegistry().apply {
            register<NotBlank> { _ -> NotBlankValidator }
            register<NotEmpty> { _ -> NotEmptyValidator }
            register<MinLength> { ann -> MinLengthValidator(ann.value) }
            register<MaxLength> { ann -> MaxLengthValidator(ann.value) }
            register<Pattern> { ann -> PatternValidator(ann.regex) }
            register<NotContains> { ann -> NotContainsValidator(ann.forbidden.toList()) }
            register<MinSize> { ann -> MinSizeValidator(ann.value) }
            register<MaxSize> { ann -> MaxSizeValidator(ann.value) }
            register<Min> { ann -> MinValidator(ann.value) }
            register<Max> { ann -> MaxValidator(ann.value) }
        }
    }
}
