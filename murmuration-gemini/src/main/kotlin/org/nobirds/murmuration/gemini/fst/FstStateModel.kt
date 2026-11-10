package org.nobirds.murmuration.gemini.fst

import com.google.genai.types.FunctionDeclaration

/**
 * Exact local model of the Vertex FST size N for a set of Gemini function declarations.
 *
 * N is the state count of the minimal DFA of the language of valid calls built from the
 * declarations, plus a constant baseline. Keys are ordered by code point, so the model reads the
 * declarations exactly as they go on the wire (after any key renaming). The property order inside
 * a schema is irrelevant: only key names decide it.
 *
 * Offline reference and source of the parity numbers: the Python reference `fstmodel.py`. The model
 * is not called at runtime; it is the single calculator for ratchets and layout planning.
 * Constraints it has no measured treatment for fail loudly, see [FstUnsupported].
 */
object FstStateModel {
    /** Constant added to the minimal DFA size (measured against Vertex). */
    const val BASELINE_STATES = 36

    /** Vertex FST size N of [declarations]. */
    fun states(declarations: List<FunctionDeclaration>): Int {
        declarations.forEach(FstUnsupported::check)
        val built = FstLanguageBuilder().build(declarations)
        val dfa = built.nfa.determinize(built.starts, built.finals)
        return dfa.minimalSize() + BASELINE_STATES
    }
}
