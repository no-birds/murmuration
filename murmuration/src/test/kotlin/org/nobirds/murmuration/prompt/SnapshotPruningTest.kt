package org.nobirds.murmuration.prompt

import org.nobirds.murmuration.prompt.CallIds
import kotlinx.coroutines.flow.flowOf
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

class SnapshotPruningTest {

    private val testLlm = Llm(
        provider = "test", family = "test", name = "test",
        version = "1", stable = true, model = "test",
    )

    private fun prompt(vararg content: PromptContent): Prompt =
        Prompt(testLlm, settings = null, content = content.toList())

    private fun user(vararg parts: PromptPart): PromptContent =
        PromptContent(PromptContentType.USER, parts.toList())

    private fun text(text: String, sectionId: String? = null, cache: PromptCacheStrategy = PromptCacheStrategy.NO_CACHE) =
        TextPromptPart(text, cacheStrategy = cache, replaceableSectionId = sectionId)

    @Test
    fun `no marked parts is a no-op`() {
        val p = prompt(
            user(text("hello"), text("world")),
            user(text("plain"))
        )
        val pruned = p.pruneReplaceableSections()
        assertEquals(p.content, pruned.content)
    }

    @Test
    fun `content with marked part is dropped whole when later content has the same id`() {
        val p = prompt(
            user(text("phase-desc-1", sectionId = "phase"), text("tools-1", sectionId = "phase")),
            user(text("between")),
            user(text("phase-desc-2", sectionId = "phase"), text("tools-2", sectionId = "phase"))
        )
        val pruned = p.pruneReplaceableSections()

        assertEquals(2, pruned.content.size)
        assertEquals(listOf("between"), pruned.content[0].parts.map { (it as TextPromptPart).text })
        assertEquals(
            listOf("phase-desc-2", "tools-2"),
            pruned.content[1].parts.map { (it as TextPromptPart).text }
        )
    }

    @Test
    fun `content drops entirely even if it carries unmarked sibling parts`() {
        // Per-content semantics: a content is the unit of replacement. If any of its ids has
        // been superseded, the whole content goes — including its non-marked parts. This is
        // intentional: tool-call / tool-result pairs ride together with their text Wrapper.
        val p = prompt(
            user(text("important context"), text("snap-v1", sectionId = "snap")),
            user(text("snap-v2", sectionId = "snap"))
        )
        val pruned = p.pruneReplaceableSections()
        assertEquals(1, pruned.content.size)
        assertEquals("snap-v2", (pruned.content[0].parts.single() as TextPromptPart).text)
    }

    @Test
    fun `content is kept only when it is latest for every id it contains`() {
        val p = prompt(
            user(text("A-1", sectionId = "A"), text("B-1", sectionId = "B")),
            user(text("A-2", sectionId = "A")),
            user(text("B-2", sectionId = "B"))
        )
        val pruned = p.pruneReplaceableSections()

        // Content 0 has both A and B; A's last is at idx 1, B's last is at idx 2 → drop.
        // Content 1 has A only; A's last is idx 1 → keep.
        // Content 2 has B only; B's last is idx 2 → keep.
        assertEquals(2, pruned.content.size)
        assertEquals(listOf("A-2"), pruned.content[0].parts.map { (it as TextPromptPart).text })
        assertEquals(listOf("B-2"), pruned.content[1].parts.map { (it as TextPromptPart).text })
    }

    @Test
    fun `marked part alone in a message drops the message entirely`() {
        val p = prompt(
            user(text("snap-1", sectionId = "snap")),
            user(text("real content")),
            user(text("snap-2", sectionId = "snap"))
        )
        val pruned = p.pruneReplaceableSections()

        assertEquals(2, pruned.content.size)
        assertEquals("real content", (pruned.content[0].parts[0] as TextPromptPart).text)
        assertEquals("snap-2", (pruned.content[1].parts[0] as TextPromptPart).text)
    }

    @Test
    fun `surviving part preserves cacheStrategy and replaceableSectionId metadata`() {
        val ttl = PromptCacheStrategy.ttl(kotlin.time.Duration.parse("PT5M"))
        val p = prompt(
            user(text("snap-1", sectionId = "snap", cache = ttl)),
            user(text("snap-2", sectionId = "snap", cache = ttl))
        )
        val pruned = p.pruneReplaceableSections()
        val survivor = pruned.content.single().parts.single() as TextPromptPart
        assertEquals("snap-2", survivor.text)
        assertEquals("snap", survivor.replaceableSectionId)
        assertEquals(ttl, survivor.cacheStrategy)
    }

    @Test
    fun `unmarked contents preserve their order across prune`() {
        val p = prompt(
            user(text("u1")),
            PromptContent(PromptContentType.MODEL, listOf(text("m1"))),
            user(text("snap-old", sectionId = "snap")),
            user(text("snap-new", sectionId = "snap"))
        )
        val pruned = p.pruneReplaceableSections()
        val types = pruned.content.map { it.type }
        assertEquals(
            listOf(PromptContentType.USER, PromptContentType.MODEL, PromptContentType.USER),
            types
        )
        val texts = pruned.content.flatMap { c -> c.parts.map { (it as TextPromptPart).text } }
        assertEquals(listOf("u1", "m1", "snap-new"), texts)
    }

    @Test
    fun `withReplaceableSectionPruning enabled false returns identity executor`() {
        val base = PromptExecutor { flowOf() }
        val decorated = base.withReplaceableSectionPruning(enabled = false)
        assertSame(base, decorated)
    }

    @Test
    fun `withReplaceableSectionPruning enabled true wraps the executor`() {
        val seen = mutableListOf<Prompt>()
        val base = PromptExecutor { p ->
            seen += p
            flowOf()
        }
        val decorated = base.withReplaceableSectionPruning(enabled = true)
        val input = prompt(
            user(text("snap-1", sectionId = "snap")),
            user(text("snap-2", sectionId = "snap"))
        )
        decorated.execute(input)
        assertEquals(1, seen.size)
        val delivered = seen.single()
        assertEquals(1, delivered.content.size)
        assertEquals("snap-2", (delivered.content.single().parts.single() as TextPromptPart).text)
    }

    @Test
    fun `prune is no-op when sections appear only once`() {
        val p = prompt(
            user(text("snap-1", sectionId = "snap")),
            user(text("tail"))
        )
        val pruned = p.pruneReplaceableSections()
        assertTrue(pruned.content.size == 2)
        assertEquals("snap-1", (pruned.content[0].parts.single() as TextPromptPart).text)
        assertEquals("tail", (pruned.content[1].parts.single() as TextPromptPart).text)
    }

    @Test
    fun `model and tool pair is dropped or kept atomically`() {
        val toolCall = ToolCallPromptPart(
            call = ToolCall(name = "addNode", args = kotlinx.serialization.json.JsonObject(emptyMap()), callId = CallIds.mint()),
            replaceableSectionId = "tool-history",
        )
        val toolResult = ToolResultPromptPart(
            name = "addNode",
            result = kotlinx.serialization.json.JsonPrimitive("ok"),
            replaceableSectionId = "tool-history",
            callId = CallIds.mint(),
        )
        val p = prompt(
            // Pair 1 — stale.
            PromptContent(PromptContentType.MODEL, listOf(toolCall)),
            PromptContent(PromptContentType.TOOL, listOf(toolResult)),
            // Pair 2 — freshest.
            PromptContent(PromptContentType.MODEL, listOf(toolCall.copy(call = ToolCall("setMetadata", kotlinx.serialization.json.JsonObject(emptyMap()), callId = CallIds.mint())))),
            PromptContent(PromptContentType.TOOL, listOf(toolResult.copy(name = "setMetadata"))),
        )
        val pruned = p.pruneReplaceableSections()
        // Pair 1 has id whose latest chunk is pair 2 → drop both contents of pair 1.
        // Pair 2 is the latest → keep both. Result: 2 contents.
        assertEquals(2, pruned.content.size)
        assertEquals(PromptContentType.MODEL, pruned.content[0].type)
        assertEquals(PromptContentType.TOOL, pruned.content[1].type)
        assertEquals("setMetadata", (pruned.content[0].parts.single() as ToolCallPromptPart).call.name)
        assertEquals("setMetadata", (pruned.content[1].parts.single() as ToolResultPromptPart).name)
    }

    @Test
    fun `model without tool result is treated as standalone`() {
        // Model message without ToolCall parts (a text reply) — NOT a pair, a standalone chunk.
        val p = prompt(
            PromptContent(PromptContentType.MODEL, listOf(text("just text", sectionId = "tool-history"))),
            PromptContent(PromptContentType.MODEL, listOf(text("newer text", sectionId = "tool-history"))),
        )
        val pruned = p.pruneReplaceableSections()
        assertEquals(1, pruned.content.size)
        assertEquals("newer text", (pruned.content.single().parts.single() as TextPromptPart).text)
    }
}
