@file:OptIn(ExperimentalSerializationApi::class)

package org.nobirds.murmuration.tooling

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialInfo
import kotlin.reflect.KClass

/**
 * SPI for setter-time parameter value transformation. Unlike [ParameterValidator] (which
 * can only accept or reject a value), a transformer **rewrites** the value before it
 * reaches the underlying `@Instrument` function.
 *
 * Typical uses: normalize escape sequences emitted by LLMs (`\\n` → `\n`), trim whitespace,
 * collapse runs of whitespace, canonicalize slugs, etc. — anything that's deterministic and
 * cheap to apply unconditionally.
 *
 * Wired in via [ParameterTransformingDecorator] which runs **before** [ParameterValidator]
 * (decorator order 5 vs 10) so validators see the post-transform value (e.g. `@NotBlank` +
 * `@Trim` rejects `"   "` correctly because trim runs first and produces `""`).
 *
 * 1. [transform] — the runtime rewrite applied before the delegate runs.
 * 2. [describe] — a short human-readable summary mixed into the parameter's schema
 *    `description` so the LLM knows the rule up-front (e.g. "literal `\n` is normalized to
 *    real newline" lets the model emit either form safely). Default empty.
 */
interface ParameterTransformer {
    fun transform(value: Any?): Any?
    fun describe(): String = ""
}

/**
 * Marks an annotation as a "transformation group" — a composing shortcut. Analog of
 * [ValidationGroup] for transformers. The runtime walks the annotations declared on the
 * marked annotation class itself and collects transformers from each of them, with cycle
 * protection. Use to build short, readable domain shortcuts:
 *
 * ```
 * @TransformationGroup
 * @Trim
 * @NormalizeNewlines
 * annotation class CleanMultiline
 * ```
 */
@Target(AnnotationTarget.ANNOTATION_CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class TransformationGroup

// --- Built-in annotations ---

/**
 * Normalize literal escape sequences emitted by LLMs back to real newlines. LLMs sometimes
 * encode a multi-line value by writing the two-character sequence `\` + `n` instead of an
 * actual newline (especially when shipping through JSON tool args that already escaped the
 * payload once). This transformer replaces `\r\n` and `\n` (as two-char sequences) with a
 * real `\n`. Real newlines already present in the input pass through untouched.
 */
@SerialInfo
@Target(AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.PROPERTY, AnnotationTarget.ANNOTATION_CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class NormalizeNewlines

/** Trim leading and trailing whitespace from a `String` parameter. No-op for non-Strings. */
@SerialInfo
@Target(AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.PROPERTY, AnnotationTarget.ANNOTATION_CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class Trim

// --- Built-in transformers ---

internal object NormalizeNewlinesTransformer : ParameterTransformer {
    override fun transform(value: Any?): Any? {
        val s = value as? String ?: return value
        return s.replace("\\r\\n", "\n").replace("\\n", "\n")
    }

    override fun describe(): String =
        "literal `\\n` / `\\r\\n` (two-char escape) is normalized to a real newline"
}

internal object TrimTransformer : ParameterTransformer {
    override fun transform(value: Any?): Any? {
        val s = value as? String ?: return value
        return s.trim()
    }

    override fun describe(): String = "leading and trailing whitespace are trimmed"
}

// --- Registry ---

/**
 * Registry of parameter-transformer factories keyed by annotation type. Analog of
 * [ValidatorRegistry]. The annotation instance is passed to the factory, which decides
 * what to extract from it.
 *
 * To add your own `@MyTransformation(...)` outside the framework:
 * 1. Declare the annotation with `@Retention(RUNTIME)` and `@Target(VALUE_PARAMETER)`.
 * 2. Register a factory: `registry.register<MyTransformation> { ann -> MyTransformer(ann.foo) }`.
 * 3. Place the annotation on a `@Instrument` parameter — the runtime assembles the
 *    transforming wrapper automatically.
 *
 * Composing — annotations marked with [TransformationGroup] are expanded recursively:
 * [resolve] walks meta-annotations on the class itself and collects all transitively
 * registered transformers (with cycle protection).
 */
class TransformerRegistry {
    @PublishedApi
    internal val factories = mutableMapOf<KClass<out Annotation>, (Annotation) -> ParameterTransformer>()

    inline fun <reified A : Annotation> register(noinline factory: (A) -> ParameterTransformer) = apply {
        @Suppress("UNCHECKED_CAST")
        factories[A::class] = factory as (Annotation) -> ParameterTransformer
    }

    /**
     * Returns a transformer for the given annotation if a factory is registered for its
     * type, `null` otherwise. Does NOT expand [TransformationGroup] composites — use
     * [resolve] for the full transitive list.
     */
    fun build(annotation: Annotation): ParameterTransformer? =
        factories[annotation.annotationClass]?.invoke(annotation)

    /**
     * Returns the full transitive list of transformers for an annotation: the transformer
     * registered for the annotation type itself (if any), plus — when the annotation is a
     * [TransformationGroup] composite — transformers for each meta-annotation on it.
     *
     * Cycle-safe: a composite that (transitively) references itself stops the walk.
     */
    fun resolve(annotation: Annotation): List<ParameterTransformer> {
        val out = mutableListOf<ParameterTransformer>()
        val visited = mutableSetOf<KClass<out Annotation>>()
        resolveInto(annotation, out, visited)
        return out
    }

    private fun resolveInto(
        annotation: Annotation,
        out: MutableList<ParameterTransformer>,
        visited: MutableSet<KClass<out Annotation>>
    ) {
        val type = annotation.annotationClass
        if (!visited.add(type)) return
        factories[type]?.invoke(annotation)?.let(out::add)
        if (type.annotations.any { it is TransformationGroup }) {
            for (meta in type.annotations) {
                if (meta is TransformationGroup) continue
                resolveInto(meta, out, visited)
            }
        }
    }

    companion object {
        /**
         * Default registry: universal transformers ([NormalizeNewlines], [Trim]).
         * Application code registers its own product-specific transformers on top.
         */
        val Default: TransformerRegistry = TransformerRegistry().apply {
            register<NormalizeNewlines> { _ -> NormalizeNewlinesTransformer }
            register<Trim> { _ -> TrimTransformer }
        }
    }
}
