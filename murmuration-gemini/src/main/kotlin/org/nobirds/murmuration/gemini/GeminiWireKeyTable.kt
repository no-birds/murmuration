package org.nobirds.murmuration.gemini

/**
 * The frozen wire key table: which field signatures sit in the tail of an object and under which code.
 * Text form (`wire-keys.tsv`): a header line `#width=<n>`, then one line `code<TAB>key<TAB>signatureHash` per
 * row, sorted by code. The code order is the tail order. Appending keeps wire names of existing fields (the provider caches
 * schemas by their text); dropping a row of a dead signature changes only the names of fields that no longer exist.
 *
 * @param width code width in characters; fixed for the life of the table (a change is a flag day)
 * @param rows rows sorted by code
 */
class GeminiWireKeyTable(val width: Int, val rows: List<Row>) {

    /** One tail field: [code], its canonical [key] (informational) and the hash of its signature. */
    data class Row(val code: Char, val key: String, val signatureHash: String)

    init {
        require(width == SUPPORTED_WIDTH) { "only one-character codes are supported, got width $width" }
        require(rows.map { it.code }.let { codes -> codes == codes.sorted() && codes.toSet().size == codes.size }) {
            "wire key table codes must be unique and sorted"
        }
        require(rows.all { it.code in GeminiWireKeyLayout.WIRE_CODE_ALPHABET }) { "wire key code must be in 0-9A-Za-z" }
        require(rows.map { it.signatureHash }.toSet().size == rows.size) { "a signature occurs twice in the wire key table" }
    }

    /** Codes of the alphabet that no row holds. */
    val freeCodes: Int get() = GeminiWireKeyLayout.WIRE_CODE_ALPHABET.length - rows.size

    fun toText(): String =
        (listOf("$HEADER$width") + rows.map { "${it.code}\t${it.key}\t${it.signatureHash}" }).joinToString("\n", postfix = "\n")

    companion object {
        /** Width of the code that follows the wire key marker; the single source (see `toCanonicalKey`). */
        const val SUPPORTED_WIDTH = 1
        private const val HEADER = "#width="

        fun parse(text: String): GeminiWireKeyTable {
            val lines = text.lines().filter { it.isNotBlank() }
            val header = requireNotNull(lines.firstOrNull()?.takeIf { it.startsWith(HEADER) }) { "wire key table has no '$HEADER' header" }
            val width = requireNotNull(header.removePrefix(HEADER).trim().toIntOrNull()) { "bad width in '$header'" }
            val rows = lines.drop(1).map { line ->
                val parts = line.split('\t')
                require(parts.size == 3 && parts[0].length == width) { "bad wire key table line '$line'" }
                Row(parts[0].single(), parts[1], parts[2])
            }
            return GeminiWireKeyTable(width, rows)
        }
    }
}
