package org.nobirds.murmuration.gemini.fst

/** Private alphabet symbols standing for classes of characters inside values. */
internal object FstSymbols {
    const val DIGIT = 1
    const val STRING_CHAR = 2
    const val ESCAPE_FOLLOW = 3
    const val HEX = 4
}
