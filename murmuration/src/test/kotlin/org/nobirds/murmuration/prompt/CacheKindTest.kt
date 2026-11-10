package org.nobirds.murmuration.prompt

import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.llm.LlmProviders
import org.nobirds.murmuration.llm.cacheKind
import org.nobirds.murmuration.tooling.CachePlacement
import kotlin.test.assertEquals

class CacheKindTest {

    private fun llm(provider: String) = Llm(
        provider = provider, family = "f", name = "n", version = "1", stable = true, model = "m-$provider",
    )

    private fun section(text: String) = PromptContent(
        PromptContentType.USER,
        listOf(TextPromptPart(text, replaceableSectionId = "snap")),
    )

    private fun prompt(llm: Llm) = Prompt(llm, null, listOf(section("snap-1"), section("snap-2")))

    private fun deliveredContentSize(llm: Llm, flag: Boolean): Int {
        val seen = mutableListOf<Prompt>()
        val decorated = PromptExecutor { p -> seen += p; flowOf() }.withReplaceableSectionPruning(flag)
        runBlocking { decorated.execute(prompt(llm)) }
        return seen.single().content.size
    }

    @Test
    fun `anthropic provider outside any catalog derives MARKS, others EXECUTOR`() {
        assertEquals(CacheKind.MARKS, llm(LlmProviders.ANTHROPIC).cacheKind)
        assertEquals(CacheKind.EXECUTOR, llm("google").cacheKind)
        assertEquals(CacheKind.EXECUTOR, llm("azure").cacheKind)
    }

    @Test
    fun `pruning decorator leaves MARKS history untouched even with the flag on`() {
        assertEquals(2, deliveredContentSize(llm(LlmProviders.ANTHROPIC), flag = true))
    }

    @Test
    fun `pruning decorator behaves as before for EXECUTOR kind`() {
        assertEquals(1, deliveredContentSize(llm("google"), flag = true))
        assertEquals(1, deliveredContentSize(llm("azure"), flag = true))
        assertEquals(2, deliveredContentSize(llm("google"), flag = false))
    }

    @Test
    fun `cache placement - tail mark only for MARKS`() {
        assertEquals(PromptCacheStrategy.NO_CACHE, CachePlacement.of(CacheKind.EXECUTOR).historyCacheStrategy)
        assertEquals(PromptCacheScope.TTL_BASED, CachePlacement.of(CacheKind.MARKS).historyCacheStrategy.scope)
    }
}
