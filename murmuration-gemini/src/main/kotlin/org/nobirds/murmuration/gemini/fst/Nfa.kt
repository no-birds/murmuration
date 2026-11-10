package org.nobirds.murmuration.gemini.fst

import java.util.BitSet

/**
 * Nondeterministic automaton with epsilon moves; symbols are Unicode code points or the private
 * value-class symbols from [FstSymbols].
 */
internal class Nfa {
    private val moves = ArrayList<HashMap<Int, MutableList<Int>>>()
    private val epsilon = ArrayList<MutableList<Int>>()

    val size: Int get() = moves.size

    fun newState(): Int {
        moves.add(HashMap())
        epsilon.add(ArrayList())
        return moves.size - 1
    }

    fun add(from: Int, symbol: Int, to: Int) {
        moves[from].getOrPut(symbol) { ArrayList(1) }.add(to)
    }

    fun addEpsilon(from: Int, to: Int) {
        epsilon[from].add(to)
    }

    /** Appends [text] code point by code point after [from]; returns the state after the last one. */
    fun literal(from: Int, text: String): Int {
        var current = from
        text.codePoints().forEach { cp ->
            val next = newState()
            add(current, cp, next)
            current = next
        }
        return current
    }

    fun closure(states: BitSet): BitSet {
        val result = states.clone() as BitSet
        val stack = ArrayDeque<Int>()
        var i = result.nextSetBit(0)
        while (i >= 0) {
            stack.addLast(i)
            i = result.nextSetBit(i + 1)
        }
        while (stack.isNotEmpty()) {
            val x = stack.removeLast()
            for (y in epsilon[x]) {
                if (!result.get(y)) {
                    result.set(y)
                    stack.addLast(y)
                }
            }
        }
        return result
    }

    /** Subset construction from [starts]; DFA state 0 is the closure of [starts]. */
    fun determinize(starts: BitSet, finals: BitSet): Dfa {
        val first = closure(starts)
        val index = HashMap<BitSet, Int>()
        val order = ArrayList<BitSet>()
        val transitions = ArrayList<HashMap<Int, Int>>()
        val dfaFinals = BitSet()
        index[first] = 0
        order.add(first)
        transitions.add(HashMap())
        var i = 0
        while (i < order.size) {
            val current = order[i]
            if (current.intersects(finals)) dfaFinals.set(i)
            val bySymbol = HashMap<Int, BitSet>()
            var x = current.nextSetBit(0)
            while (x >= 0) {
                for ((symbol, targets) in moves[x]) {
                    val set = bySymbol.getOrPut(symbol) { BitSet() }
                    for (t in targets) set.set(t)
                }
                x = current.nextSetBit(x + 1)
            }
            for ((symbol, targets) in bySymbol) {
                val target = closure(targets)
                val id = index.getOrPut(target) {
                    order.add(target)
                    transitions.add(HashMap())
                    order.size - 1
                }
                transitions[i][symbol] = id
            }
            i++
        }
        return Dfa(transitions, dfaFinals)
    }
}

/** Deterministic automaton; state 0 is the start state. */
internal class Dfa(val transitions: List<Map<Int, Int>>, val finals: BitSet) {
    val size: Int get() = transitions.size

    /** The automaton of the reversed language: starts are this DFA's finals, the only final is its start. */
    fun reversed(): ReversedDfa {
        val nfa = Nfa()
        repeat(size) { nfa.newState() }
        transitions.forEachIndexed { from, row -> for ((symbol, to) in row) nfa.add(to, symbol, from) }
        return ReversedDfa(nfa, finals.clone() as BitSet, BitSet().also { it.set(0) })
    }

    /** Minimal DFA state count by Brzozowski: reverse and determinize twice. */
    fun minimalSize(): Int = reversed().determinize().reversed().determinize().size
}

internal class ReversedDfa(val nfa: Nfa, val starts: BitSet, val finals: BitSet) {
    fun determinize(): Dfa = nfa.determinize(starts, finals)
}
