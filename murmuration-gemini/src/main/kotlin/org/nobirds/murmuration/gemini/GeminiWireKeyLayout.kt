package org.nobirds.murmuration.gemini

/**
 * Which argument fields move to the tail of the function-declaration schema and under which wire name.
 *
 * The table is keyed by field signature (see [FieldSignature]), not by tool or profile: a field whose
 * signature is absent keeps its canonical name. A wire name is [WIRE_KEY_MARKER] + one code char + the
 * canonical key, so it can be reversed lexically (see `toCanonical`). With no entries the layout is the
 * identity and the provider sees canonical schemas.
 *
 * @param codeByHash signature hash -> one-character code (`0-9A-Za-z`), codes unique; the frozen table stores
 * hashes, not signatures (build from signatures with [ofSignatures], from the table with [of])
 */
class GeminiWireKeyLayout private constructor(private val codeByHash: Map<String, Char>) {

    init {
        require(codeByHash.values.toSet().size == codeByHash.size) { "wire key codes must be unique" }
        require(codeByHash.values.all { it in WIRE_CODE_ALPHABET }) { "wire key code must be in 0-9A-Za-z" }
    }

    /** Wire name for the field with [signature] and canonical [key]; `null` when the field stays canonical. */
    fun wireKeyOf(signature: String, key: String): String? =
        if (codeByHash.isEmpty()) null else codeByHash[FieldSignature.hash(signature)]?.let { "$WIRE_KEY_MARKER$it$key" }

    companion object {
        /** Identity layout: every key stays canonical. Written explicitly wherever a canonical schema is intended. */
        val NONE = GeminiWireKeyLayout(emptyMap())

        /** Layout from field signatures (tests and planners that work with signatures, not the frozen table). */
        fun ofSignatures(codeBySignature: Map<String, Char>): GeminiWireKeyLayout =
            GeminiWireKeyLayout(codeBySignature.mapKeys { FieldSignature.hash(it.key) })

        /** Layout of the frozen [table]; its rows hold signature hashes, which is what the layout is keyed by. */
        fun of(table: GeminiWireKeyTable): GeminiWireKeyLayout =
            GeminiWireKeyLayout(table.rows.associate { it.signatureHash to it.code })

        /** The only alphabet of wire codes, in code order (the tail order of a table is the order of these chars). */
        internal const val WIRE_CODE_ALPHABET: String = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    }
}
