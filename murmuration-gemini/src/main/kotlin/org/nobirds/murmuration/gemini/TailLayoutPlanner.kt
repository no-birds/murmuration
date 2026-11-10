package org.nobirds.murmuration.gemini

import com.google.genai.types.FunctionDeclaration

/**
 * Offline chooser of the tail of the wire layout: which field signatures move to the end of their objects and
 * in what order. Port of the offline Python reference search (`union_search`), deterministic:
 * profiles go by name, candidates by total count (stable over first-seen order), a move is accepted only on a
 * strict improvement.
 *
 * One table serves all profiles, so the objective is the sum of [ProfileObjective] proxies over the profiles
 * that hold the moved signature. Candidates are signatures that occur at least twice in some profile.
 * The planner runs in the table generator only; runtime applies the frozen table.
 *
 * @param profiles tool declarations of every mounted profile, by profile name
 * @param rounds local-search rounds
 */
class TailLayoutPlanner(profiles: Map<String, List<FunctionDeclaration>>, private val rounds: Int = 3) {

    private val objectives: List<ProfileObjective>

    init {
        val weights = FieldWeights()
        objectives = profiles.toSortedMap().values.map { ProfileObjective.of(it, weights) }
    }

    /** Signatures to move to the tail, earliest tail element first. */
    fun plan(): List<String> {
        val candidates = candidates()
        val holders = candidates.associateWith { s -> objectives.filter { s in it.frequency } }
        fun cost(order: List<String>, signature: String): Int {
            val rank = order.withIndex().associate { (i, s) -> s to i }
            return holders.getValue(signature).sumOf { it.cost(rank) }
        }

        var order = emptyList<String>()
        for (s in candidates) {
            val trial = listOf(s) + order
            if (cost(trial, s) < cost(order, s)) order = trial
        }
        for (s in order.toList()) {
            val trial = order - s
            if (cost(trial, s) < cost(order, s)) order = trial
        }
        repeat(rounds) {
            var improved = false
            for (s in candidates) {
                val without = order - s
                var current = cost(order, s)
                for (position in 0..without.size) {
                    val trial = without.take(position) + s + without.drop(position)
                    val c = cost(trial, s)
                    if (c < current) { current = c; order = trial; improved = true }
                }
                if (s in order) {
                    val c = cost(without, s)
                    if (c < current) { current = c; order = without; improved = true }
                }
            }
            if (!improved) return order
        }
        return order
    }

    /**
     * The table after one update: rows of dead signatures ([staleRows]) dropped, then [extend]. Dropping is safe:
     * the history of cycles lives in the process memory only and `toCanonical` is lexical, so a changed
     * "code -> signature" correspondence costs one re-creation of provider caches, not correctness
     *. Appending-only is an optimisation (stable wire names, a readable git diff),
     * not an invariant. The table is the fixed point of this function when it is up to date.
     */
    fun update(table: GeminiWireKeyTable): GeminiWireKeyTable {
        val dead = staleRows(table)
        return extend(if (dead.isEmpty()) table else GeminiWireKeyTable(table.width, table.rows - dead.toSet()))
    }

    /**
     * Appends to [table] every new candidate signature that lowers the objective, in the free code between its
     * neighbours (insertion between rows keeps every existing code; no free code at a position means no insertion
     * there). Existing rows never change or move. Returns [table] itself when nothing is worth appending.
     */
    fun extend(table: GeminiWireKeyTable): GeminiWireKeyTable {
        val signatureByHash = candidatesAndAll().associateBy { FieldSignature.hash(it) }
        var rows: List<GeminiWireKeyTable.Row> = table.rows
        val signatureOf = { row: GeminiWireKeyTable.Row -> signatureByHash[row.signatureHash] }
        val known = rows.mapNotNullTo(HashSet()) { signatureOf(it) }
        val holders = candidates().associateWith { s -> objectives.filter { s in it.frequency } }
        fun cost(trial: List<GeminiWireKeyTable.Row>, signature: String): Int {
            val rank = HashMap<String, Int>()
            trial.forEachIndexed { i, row -> signatureOf(row)?.let { rank[it] = i } }
            return holders.getValue(signature).sumOf { it.cost(rank) }
        }
        for (s in candidates()) {
            if (s in known) continue
            var best = cost(rows, s)
            var bestRows: List<GeminiWireKeyTable.Row>? = null
            for (position in 0..rows.size) {
                val low = if (position == 0) -1 else GeminiWireKeyLayout.WIRE_CODE_ALPHABET.indexOf(rows[position - 1].code)
                val high = if (position == rows.size) GeminiWireKeyLayout.WIRE_CODE_ALPHABET.length else GeminiWireKeyLayout.WIRE_CODE_ALPHABET.indexOf(rows[position].code)
                if (high - low < 2) continue
                val row = GeminiWireKeyTable.Row(GeminiWireKeyLayout.WIRE_CODE_ALPHABET[(low + high) / 2], s.takeWhile { it != '!' && it != '?' }, FieldSignature.hash(s))
                val trial = rows.take(position) + row + rows.drop(position)
                val c = cost(trial, s)
                if (c < best) { best = c; bestRows = trial }
            }
            if (bestRows != null) { rows = bestRows; known += s }
        }
        return if (rows == table.rows) table else GeminiWireKeyTable(table.width, rows)
    }

    /**
     * The table a from-scratch plan would give: [plan] order, codes spread over the whole alphabet ([codesOf]).
     * Never written to production: it is the optimum that the frozen table is compared against (staleness signal);
     * adopting it changes wire names of existing fields, so it is a flag day.
     */
    fun repack(): GeminiWireKeyTable {
        val order = plan()
        val codes = codesOf(order)
        val rows = order.map { s -> GeminiWireKeyTable.Row(codes.getValue(s), s.takeWhile { it != '!' && it != '?' }, FieldSignature.hash(s)) }
        return GeminiWireKeyTable(GeminiWireKeyTable.SUPPORTED_WIDTH, rows.sortedBy { it.code })
    }

    /** Rows of [table] whose signature occurs in no profile: the schema moved on, the row is dead. */
    fun staleRows(table: GeminiWireKeyTable): List<GeminiWireKeyTable.Row> {
        val live = candidatesAndAll().mapTo(HashSet()) { FieldSignature.hash(it) }
        return table.rows.filter { it.signatureHash !in live }
    }

    /** Every distinct signature of every profile, field-blocked ones excluded as in [ProfileObjective.frequency]. */
    private fun candidatesAndAll(): Set<String> = objectives.flatMapTo(LinkedHashSet()) { it.frequency.keys }

    /** Signatures with count >= 2 in some profile, by total count over profiles descending (ties: first seen). */
    private fun candidates(): List<String> {
        val seen = LinkedHashSet<String>()
        val total = HashMap<String, Int>()
        for (objective in objectives) {
            objective.frequency.entries.sortedByDescending { it.value }.forEach { (s, c) -> if (c >= 2) seen += s }
            objective.frequency.forEach { (s, c) -> total.merge(s, c, Int::plus) }
        }
        return seen.sortedByDescending { total.getValue(it) }
    }

    companion object {
        
        /**
         * Codes for [order], spread over the whole alphabet so the holes between neighbours stay free for later
         * insertions (`union.py:wire_codes` with equal step). Needs at most 62 signatures.
         */
        fun codesOf(order: List<String>): Map<String, Char> {
            require(order.size <= GeminiWireKeyLayout.WIRE_CODE_ALPHABET.length) { "${order.size} signatures do not fit ${GeminiWireKeyLayout.WIRE_CODE_ALPHABET.length} one-char codes" }
            return order.withIndex().associate { (i, s) -> s to GeminiWireKeyLayout.WIRE_CODE_ALPHABET[(i * GeminiWireKeyLayout.WIRE_CODE_ALPHABET.length) / order.size] }
        }
    }
}
