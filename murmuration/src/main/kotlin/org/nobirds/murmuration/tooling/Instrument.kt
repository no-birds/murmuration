package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.agent.*
import org.nobirds.murmuration.prompt.*
import org.nobirds.murmuration.llm.*

import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import java.util.UUID
import kotlin.reflect.KCallable
import kotlin.reflect.KClass
import kotlin.reflect.KFunction
import kotlin.reflect.KParameter
import kotlin.reflect.KType
import kotlin.reflect.full.allSuperclasses
import kotlin.reflect.full.callSuspendBy
import kotlin.reflect.full.findAnnotation
import kotlin.reflect.full.functions
import kotlin.reflect.full.memberProperties
import kotlin.reflect.jvm.isAccessible
import kotlin.reflect.jvm.javaMethod


/**
 * Marks a function, class, or property as an LLM instrument.
 * @param description Description of the instrument exposed to the LLM.
 * @param name Optional override for the instrument name (defaults to the function name).
 *
 * Other aspects (phase availability, restrictions, validation) live in separate annotations
 * wired in via [InstrumentDecorator]s.
 */
@Target(AnnotationTarget.FUNCTION, AnnotationTarget.CLASS, AnnotationTarget.PROPERTY)
@Retention(AnnotationRetention.RUNTIME)
annotation class Instrument(
    val description: String = "",
    val name: String = "",
)

/**
 * Describes a single parameter of an instrument.
 *
 * If you hit Vertex Gemini's `Constraint is too tall: <N> vs max of 5888` on a tool schema,
 * see the "Keeping Gemini function-calling schemas small" section of `docs/tools-and-schemas.md`
 * in the repository. Short version: the biggest contributors in our experience are long enum values
 * (use `@SerialName("XX")` aliasing) and unused `maxLength`/numeric bounds on
 * `@SchemaInfo` (drop them, move the recommendation into the description text). The
 * obvious `optional → required+nullable` trick (dropping the Kotlin `= null` default) does
 * NOT help on Vertex — its FST treats both forms the same.
 */
@Target(AnnotationTarget.VALUE_PARAMETER)
@Retention(AnnotationRetention.RUNTIME)
annotation class InstrumentParam(val description: String = "")

/**
 * Marks a function as a framework-internal hook that must NOT be exposed as an LLM tool,
 * even when its declaring class carries a class-level [Instrument]. Use this on methods of
 * framework protocols (e.g. lifecycle callbacks on [Stateful]) so that user code with
 * `@Instrument` on the class can implement them without the factory trying to turn the
 * hook into a tool — which would fail because hook parameters do not carry
 * [InstrumentParam] descriptions.
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
annotation class NotAnInstrument

/**
 * Builds [InstrumentsDeclaration]s out of plain Kotlin objects. The wrapping chain
 * (parameter validation, cross-param validation, task-closing) is plugged in through a
 * [DecoratorRegistry], which can be customised by passing your own. Defaults to
 * [DecoratorRegistry.Default].
 */
class InstrumentFactory(
    private val decorators: DecoratorRegistry = DecoratorRegistry.Default,
) {

    fun <T : Any> create(obj: T): InstrumentsDeclaration<T> {
        val klass = obj::class
        val instruments = mutableListOf<InstrumentDeclaration<*>>()

        // 1. The object itself may be an InstrumentDeclaration.
        if (obj is InstrumentDeclaration<*>) {
            instruments.add(obj)
        }

        val isAnnotated = { it: KClass<*> -> it.findAnnotation<Instrument>() != null }
        val classAnnotated = isAnnotated(klass) || klass.allSuperclasses.any(isAnnotated)

        // 2. Public functions, marked with @Instrument or implicitly via a class-level @Instrument.
        // Functions annotated `@NotAnInstrument` (framework hooks like Stateful.onAfterToolCalls)
        // are always skipped — even via class-level @Instrument — because their parameters do
        // not carry @InstrumentParam.
        klass.functions
            .filter { function ->
                val allAnnotations = function.collectAllAnnotations()
                if (allAnnotations.any { it is NotAnInstrument }) return@filter false
                allAnnotations.any { it is Instrument } ||
                    (classAnnotated && function.visibility == kotlin.reflect.KVisibility.PUBLIC &&
                        !listOf("equals", "hashCode", "toString").contains(function.name))
            }
            .sortedBy { it.name }
            .forEach { function ->
                val raw = KFunctionInstrumentDeclaration(obj, function)
                instruments.add(decorators.apply(raw, obj))
            }

        // 3. Properties marked with @Instrument that hold InstrumentDeclaration instances.
        klass.memberProperties
            .filter { it.findAnnotation<Instrument>() != null }
            .sortedBy { it.name }
            .forEach { property ->
                val value = try { property.call(obj) } catch (_: Exception) { null }
                if (value is InstrumentDeclaration<*>) {
                    instruments.add(value)
                }
            }

        return InstrumentsDeclaration.of(obj, instruments)
    }
}

/**
 * Reflection-backed [InstrumentDeclaration]. Annotations (on the function and on its
 * parameters) are merged across inheritance: an `@Annotation` placed on a same-named
 * method or parameter of a super-class is included in [annotations]. This keeps the
 * expected behaviour for `@Instrument` / `@InstrumentParam` / `@CantCloseTask` declared
 * on interfaces and used through implementations.
 */
class KFunctionInstrumentDeclaration<R>(
    override val obj: Any,
    val function: KFunction<R>,
) : InstrumentDeclaration<R> {
    override val annotations: List<Annotation> = function.collectAllAnnotations()

    private val instrumentAnnotation: Instrument? = annotations.filterIsInstance<Instrument>().firstOrNull()

    override val name: String = function.instrumentName()
    override val description: String = instrumentAnnotation?.description ?: ""

    override val parameters: List<InstrumentParameter> = function.parameters
        .filter { it.kind == KParameter.Kind.VALUE }
        .map { parameter ->
            val paramAnnotations = parameter.collectAllAnnotations(function)
            val description = paramAnnotations.filterIsInstance<InstrumentParam>().firstOrNull()?.description
                ?: throw IllegalArgumentException("Parameter '${parameter.name}' of function '${function.name}' must have @InstrumentParam annotation with description")

            InstrumentParameter(
                name = parameter.name ?: throw IllegalArgumentException("Parameter of function ${function.name} must have a name"),
                type = parameter.type,
                description = description,
                isOptional = parameter.isOptional,
                annotations = paramAnnotations,
            )
        }

    override val returnType: KType = function.returnType

    /** Parameters declared as [JsonElement]: they receive the exact wire value, never decoded. */
    private val rawJsonParameters: Set<String> =
        parameters.filter { it.type.classifier == JsonElement::class }.map { it.name }.toSet()

    /**
     * The one decoder of every other parameter. Built here, i.e. at mount ([InstrumentFactory.create]),
     * so a case-ambiguous enum anywhere in the parameters fails mounting.
     */
    private val argumentDecoder = ToolArgumentDecoder(
        parameters.filter { it.name !in rawJsonParameters }.map { it.name to it.type },
    )

    /**
     * Decodes the provider's raw arguments ONCE, at the edge: [JsonElement] parameters get the exact
     * wire value, every other declared parameter its typed value through [ToolArgumentDecoder]
     * (which throws [ToolArgumentException] with all problems of the call). Keys the function does
     * not declare are passed on as [JsonElement.toAny] and ignored by [invoke].
     */
    override fun prepareWireArguments(args: Map<String, JsonElement>): Map<String, Any?> {
        val decoded = argumentDecoder.decode(args)
        return args.mapValues { (name, value) ->
            when (name) {
                in rawJsonParameters -> value
                in argumentDecoder.parameterNames -> decoded[name]
                else -> value.toAny()
            }
        }
    }

    override suspend fun invoke(args: Map<String, Any?>): R {
        function.isAccessible = true
        val params = mutableMapOf<KParameter, Any?>()

        function.parameters.forEach { parameter ->
            when (parameter.kind) {
                KParameter.Kind.INSTANCE -> params[parameter] = obj
                KParameter.Kind.VALUE -> {
                    val name = parameter.name ?: return@forEach
                    if (args.containsKey(name)) {
                        params[parameter] = coerceValue(args[name], parameter.type)
                    } else {
                        if (parameter.isOptional) {
                            // Leave unset for callBy to fill in the default.
                        } else if (parameter.type.isMarkedNullable) {
                            params[parameter] = null
                        } else {
                            // For a suspend function the last parameter is the synthetic
                            // Continuation — it must not be present in args.
                            if (!function.isSuspend || parameter.index < function.parameters.size - 1) {
                                throw IllegalArgumentException("Missing required parameter $name for function ${function.name}")
                            }
                        }
                    }
                }
                else -> {}
            }
        }

        return if (function.isSuspend) {
            function.callSuspendBy(params)
        } else {
            function.callBy(params)
        }
    }

    private fun coerceValue(value: Any?, targetType: KType): Any? {
        val targetClass = targetType.classifier as? KClass<*> ?: return value
        if (value == null) {
            return if (targetClass == JsonElement::class) JsonNull else null
        }

        if (value is Number) {
            if (targetClass == Int::class) return value.toInt()
            if (targetClass == Long::class) return value.toLong()
            if (targetClass == Double::class) return value.toDouble()
            if (targetClass == Float::class) return value.toFloat()
            if (targetClass == Short::class) return value.toShort()
            if (targetClass == Byte::class) return value.toByte()
        }

        if (targetClass == String::class) return value.toString()

        if (targetClass == UUID::class && value is String) {
            return UUID.fromString(value)
        }

        if (targetClass == JsonElement::class) {
            return value.toJsonElement()
        }

        if (targetClass.isInstance(value)) return value

        return value
    }
}

/**
 * THE rule that names a tool, shared by everything that has to agree on the name: the schema
 * ([KFunctionInstrumentDeclaration.name] — what the model sees and calls), the name of the
 * current call ([currentToolName]) and any code that maps a function/property reference to its
 * tool. `@Instrument(name = ...)` wins; otherwise the function (or property) name. For a function
 * the annotation is looked up with inheritance merged in, exactly like [collectAllAnnotations]
 * does for the declaration.
 *
 * A Kotlin name is NOT the tool name by itself: it coincides with it only while no tool uses
 * `@Instrument(name = ...)`. Call this function instead of reading `KCallable.name`.
 */
fun KCallable<*>.instrumentName(): String {
    val declared = if (this is KFunction<*>) collectAllAnnotations() else annotations
    return declared.filterIsInstance<Instrument>().firstOrNull()?.name?.takeIf { it.isNotEmpty() } ?: name
}

/**
 * Collects all annotations of this function plus annotations of same-named methods (with
 * matching arity) from super-classes. Important for annotations declared on an interface
 * (`@Instrument`, `@CantCloseTask`) and implemented by a sub-class.
 *
 * Public (not `internal`) so that consumers scanning tool declarations call INTO this merge
 * algorithm instead of keeping a hand-copied duplicate: two copies of the SAME merge rule can drift
 * apart, so the rule lives in exactly one place.
 */
fun KFunction<*>.collectAllAnnotations(): List<Annotation> {
    val result = mutableListOf<Annotation>()
    result.addAll(annotations)

    val javaMethod = javaMethod ?: return result
    val klass = javaMethod.declaringClass.kotlin

    for (superClass in klass.allSuperclasses) {
        val superMethod = superClass.functions.find { it.name == name && it.parameters.size == parameters.size } ?: continue
        result.addAll(superMethod.annotations)
    }

    return result
}

/**
 * Same idea as [collectAllAnnotations], but for a single parameter — also collects
 * annotations from same-named parameters of same-named super-class methods.
 */
internal fun KParameter.collectAllAnnotations(function: KFunction<*>): List<Annotation> {
    val result = mutableListOf<Annotation>()
    result.addAll(annotations)

    val javaMethod = function.javaMethod ?: return result
    val klass = javaMethod.declaringClass.kotlin

    for (superClass in klass.allSuperclasses) {
        val superMethod = superClass.functions.find { it.name == function.name && it.parameters.size == function.parameters.size } ?: continue
        val superParam = superMethod.parameters.find { it.index == index } ?: continue
        result.addAll(superParam.annotations)
    }

    return result
}
