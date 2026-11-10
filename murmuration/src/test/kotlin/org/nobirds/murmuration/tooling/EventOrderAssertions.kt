package org.nobirds.murmuration.tooling

import org.nobirds.murmuration.prompt.PromptExecutionEvent
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Event-order contract of a provider executor: the SAME
 * helper shape is kept in each module's tests (no shared test fixtures between modules).
 *
 * [assertPromptStartedFirstAndOnce]: exactly one `PromptStarted`, and nothing but `CacheCreated`
 * (a side-channel event) precedes it.
 */
internal fun assertPromptStartedFirstAndOnce(events: List<PromptExecutionEvent>) {
    val started = events.indices.filter { events[it] is PromptExecutionEvent.PromptStarted }
    assertEquals(1, started.size, "exactly one PromptStarted per round-trip; events=${events.names()}")
    val before = events.take(started.single()).filterNot { it is PromptExecutionEvent.CacheCreated }
    assertTrue(before.isEmpty(), "PromptStarted must come first; preceded by ${before.names()}")
}

/**
 * [assertStreamRespondedOnce]: exactly one `StreamResponded`, after `PromptStarted` and before the
 * first `PartReceived`. Matched by class NAME (kept from step S1, when the event did not exist yet).
 */
internal fun assertStreamRespondedOnce(events: List<PromptExecutionEvent>) {
    val responded = events.indices.filter { events[it]::class.simpleName == "StreamResponded" }
    assertEquals(1, responded.size, "exactly one StreamResponded per round-trip; events=${events.names()}")
    val startedAt = events.indexOfFirst { it is PromptExecutionEvent.PromptStarted }
    assertTrue(startedAt in 0 until responded.single(), "StreamResponded must follow PromptStarted; events=${events.names()}")
    // Nothing but `PromptStarted` may precede it: usage/parts arriving first would mean the provider
    // answered before the event that says so.
    val between = events.subList(startedAt + 1, responded.single())
    assertTrue(between.isEmpty(), "StreamResponded must come right after PromptStarted; in between ${between.names()}")
    val firstPart = events.indexOfFirst { it is PromptExecutionEvent.PartReceived }
    assertTrue(firstPart == -1 || responded.single() < firstPart, "StreamResponded must precede the first PartReceived; events=${events.names()}")
}

/** The count of `StreamResponded` events, by class name (see [assertStreamRespondedOnce]). */
internal fun List<PromptExecutionEvent>.streamRespondedCount(): Int = count { it::class.simpleName == "StreamResponded" }

internal fun List<PromptExecutionEvent>.names(): List<String?> = map { it::class.simpleName }
