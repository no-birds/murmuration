package org.nobirds.murmuration.gemini

import com.google.genai.types.Schema
import com.google.genai.types.Type
import java.security.MessageDigest

/**
 * Field signature: what the model can tell apart about an object field. Two fields with the same
 * signature are interchangeable for the layout and get the same wire code. Port of `layout.py:fsig`.
 * Computed on canonical schemas, before any renaming.
 */
internal object FieldSignature {

    /** Signature of property [key] of the OBJECT schema [parent]: key, required flag, value shape. */
    fun of(parent: Schema, key: String): String =
        key + requiredFlag(parent, key) + shape(parent.properties().orElse(emptyMap()).getValue(key))

    /** A field inside an object shape: like [of] but with a `:` between the required flag and the value shape. */
    private fun nested(parent: Schema, key: String): String {
        val value = parent.properties().orElse(emptyMap()).getValue(key)
        return key + requiredFlag(parent, key) + ":" + shape(value)
    }

    private fun requiredFlag(parent: Schema, key: String): String =
        if (key in parent.required().orElse(emptyList())) "!" else "?"

    /** Stable 16-hex-char id of a signature (SHA-256 prefix); the same as `union.py:sig_hash`, used in `wire-keys.tsv`. */
    fun hash(signature: String): String =
        MessageDigest.getInstance("SHA-256").digest(signature.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it) }.take(16)

    private fun shape(schema: Schema): String {
        val nullable = if (schema.nullable().orElse(false)) "~" else ""
        schema.anyOf().orElse(null)?.let { variants ->
            return "anyOf(" + variants.joinToString("|") { shape(it) } + ")" + nullable
        }
        val type = schema.type().orElse(null)?.knownEnum()
        return when {
            type == Type.Known.OBJECT -> {
                val properties = schema.properties().orElse(emptyMap())
                "{" + properties.keys.sorted().joinToString(",") { nested(schema, it) } + "}" + nullable
            }
            type == Type.Known.ARRAY -> {
                val items = schema.items().map { shape(it) }.orElse("None")
                "[" + items + (if (schema.minItems().orElse(0L) > 0) "+" else "") + "]" + bounds(schema) + nullable
            }
            type == Type.Known.STRING && schema.enum_().orElse(emptyList()).isNotEmpty() ->
                "E(" + schema.enum_().get().joinToString("|") + ")" + nullable
            else -> when (type) {
                Type.Known.STRING -> "s"
                Type.Known.INTEGER -> "i"
                Type.Known.NUMBER -> "n"
                Type.Known.BOOLEAN -> "b"
                else -> type?.name ?: "None"
            } + bounds(schema) + nullable
        }
    }

    private fun bounds(schema: Schema): String = buildString {
        schema.format().ifPresent { append("<format=").append(it).append(">") }
        schema.minimum().ifPresent { append("<minimum=").append(it).append(">") }
        schema.maximum().ifPresent { append("<maximum=").append(it).append(">") }
    }
}
