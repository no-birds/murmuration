package org.nobirds.murmuration.gemini

import com.google.genai.types.Schema
import org.nobirds.murmuration.prompt.InstrumentDeclaration
import org.nobirds.murmuration.prompt.InstrumentsDeclaration
import org.nobirds.murmuration.prompt.ToolsPromptSettings
import org.nobirds.murmuration.tooling.InstrumentFactory

/** Test-only layout table: every distinct field signature of the given schemas gets the next code. */
internal object WireLayoutTestSupport {

    fun canonicalSchemas(instruments: List<InstrumentDeclaration<*>>): List<Schema> =
        GeminiToolDeclarations.of(instruments, GeminiWireKeyLayout.NONE).map { it.parameters().get() }

    /** Layout that moves every field of [schemas] (all signatures) to the tail; codes `a`, `b`, ... */
    fun layoutOf(schemas: List<Schema>): GeminiWireKeyLayout {
        val signatures = linkedSetOf<String>()
        fun walk(schema: Schema) {
            schema.properties().orElse(emptyMap()).forEach { (key, value) ->
                signatures += FieldSignature.of(schema, key)
                walk(value)
            }
            schema.items().ifPresent { walk(it) }
            schema.anyOf().ifPresent { variants -> variants.forEach { walk(it) } }
        }
        schemas.forEach { walk(it) }
        val alphabet = ('a'..'z').toList() + ('A'..'Z') + ('0'..'9')
        return GeminiWireKeyLayout.ofSignatures(signatures.withIndex().associate { (i, sig) -> sig to alphabet[i] })
    }

    fun instrumentsOf(tools: Any): List<InstrumentDeclaration<*>> = InstrumentFactory().create(tools).instruments

    fun settingsOf(tools: Any): ToolsPromptSettings =
        ToolsPromptSettings(listOf(InstrumentsDeclaration.of(tools, instrumentsOf(tools))))
}
