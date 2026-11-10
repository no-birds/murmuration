package org.nobirds.murmuration.gemini

import com.google.genai.Client
import com.google.genai.types.Content
import com.google.genai.types.CreateCachedContentConfig
import com.google.genai.types.FunctionCallingConfig
import com.google.genai.types.FunctionCallingConfigMode
import com.google.genai.types.ToolConfig
import com.google.genai.types.UpdateCachedContentConfig
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.future.await
import org.nobirds.murmuration.gemini.internal.logger
import org.nobirds.murmuration.gemini.internal.sha256
import org.nobirds.murmuration.prompt.FileDataPromptPart
import org.nobirds.murmuration.prompt.PromptCacheScope
import org.nobirds.murmuration.prompt.PromptContent
import org.nobirds.murmuration.prompt.PromptContentType
import org.nobirds.murmuration.prompt.PromptPart
import org.nobirds.murmuration.prompt.PromptSettings
import org.nobirds.murmuration.prompt.TextPromptPart
import org.nobirds.murmuration.prompt.ToolCallPromptPart
import org.nobirds.murmuration.prompt.ToolResultPromptPart
import org.nobirds.murmuration.prompt.ToolsPromptSettings
import org.nobirds.murmuration.prompt.cache.CacheBackend
import org.nobirds.murmuration.prompt.cache.CreateCacheResult
import org.nobirds.murmuration.prompt.dump
import org.nobirds.murmuration.prompt.dumpSettings
import org.nobirds.murmuration.retry.DefaultRetryStrategy
import org.nobirds.murmuration.retry.RetryPolicy
import org.nobirds.murmuration.retry.retryWithTimeout
import kotlin.time.Duration
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Instant

private val CACHE_OPERATION_TIMEOUT = 3.minutes

/** Retry policy of cache create/refresh: the framework default (quota ladder, separate budgets) over the Gemini strategy. */
private val CACHE_RETRY_POLICY = RetryPolicy(strategy = DefaultRetryStrategy or GeminiRetryStrategy, maxBackoff = 1.minutes)

/**
 * Gemini-specific implementation of `CacheBackend`. Only I/O against the Gemini Context Cache API
 * and converting our `PromptPart`/`PromptSettings` into native `Content`/`Tool` for the
 * Gemini SDK. No "when to cache / how to look up" logic — that lives in `CacheManager`.
 */
class GeminiCacheBackend(
    private val client: Client,
    private val wireKeyLayout: GeminiWireKeyLayout,
    private val retryPolicy: RetryPolicy = CACHE_RETRY_POLICY,
) : CacheBackend {
    private val logger = logger()

    override suspend fun createCache(
        model: String,
        contents: List<PromptContent>,
        settings: PromptSettings?,
        hash: String,
        ttl: Duration,
        scope: PromptCacheScope,
    ): CreateCacheResult = retryWithTimeout(
        timeoutPerAttempt = CACHE_OPERATION_TIMEOUT,
        operationName = "Cache create [$hash]",
        policy = retryPolicy,
    ) {
        // Declarations and cached history come from one build: one wire key layout for both.
        val wireDeclarations = GeminiToolDeclarations.build(
            (settings as? ToolsPromptSettings)?.tools?.flatMap { it.instruments }.orEmpty(),
            wireKeyLayout,
        )
        val converter = wireDeclarations.historyConverter
        val systemInstruction = converter.toSystemInstruction(
            contents.filter { it.type == PromptContentType.SYSTEM }.flatMap { it.parts }
        )
        val genaiContents = converter.toGenaiContents(contents.filter { it.type != PromptContentType.SYSTEM })

        val cacheConfigBuilder = CreateCachedContentConfig.builder()
            .contents(genaiContents)
            .displayName("Prompt Cache $hash")
            .ttl(java.time.Duration.ofMillis(ttl.inWholeMilliseconds))

        if (systemInstruction != null) cacheConfigBuilder.systemInstruction(systemInstruction)

        if (settings is ToolsPromptSettings) {
            val tool = wireDeclarations.tool
            if (tool != null) {
                cacheConfigBuilder.tools(listOf(tool))
                cacheConfigBuilder.toolConfig(
                    ToolConfig.builder()
                        .functionCallingConfig(
                            FunctionCallingConfig.builder()
                                .mode(FunctionCallingConfigMode("ANY"))
                                .build()
                        )
                        .build()
                )
            }
        }

        val totalSize = contents.flatMap { it.parts }.sumOf { estimateSize(it) }
        logger.info("Creating new Context Cache for hash: $hash, model: $model, scope: $scope, size: $totalSize bytes, ttl: $ttl")
        // Full dump of the cache contents — TRACE: the content is heavy and is already in the DB via
        // `llm_statistics`. INFO above already shows hash/model/scope/size/ttl.
        if (logger.isTraceEnabled) {
            logger.trace(
                "Cache create payload [hash={}, settings={}]\n{}",
                hash, settings.dumpSettings(), contents.dump("Cache contents"),
            )
        }

        val response = quotaAware { client.async.caches.create(model, cacheConfigBuilder.build()).await() }
        val cacheId = response.name().get()
        val expireTime = Instant.parse(response.expireTime().get().toString())
        val tokenCount = response.usageMetadata().get().totalTokenCount().get().toLong()
        logger.debug("Created Context Cache id={} for hash={} (tokens={}, expires={})", cacheId, hash, tokenCount, expireTime)

        CreateCacheResult(cacheId = cacheId, tokenCount = tokenCount, expirationTime = expireTime)
    }

    override suspend fun refreshCache(cacheId: String, ttl: Duration): Instant = retryWithTimeout(
        timeoutPerAttempt = CACHE_OPERATION_TIMEOUT,
        operationName = "Cache refresh [$cacheId]",
        policy = retryPolicy,
    ) {
        val updateConfig = UpdateCachedContentConfig.builder()
            .ttl(java.time.Duration.ofMillis(ttl.inWholeMilliseconds))
            .build()
        val response = quotaAware { client.async.caches.update(cacheId, updateConfig).await() }
        Instant.parse(response.expireTime().get().toString())
    }

    /** A raw genai 429 becomes [org.nobirds.murmuration.prompt.QuotaExceededException] (same mapper as the executor), so the quota ladder of the retry applies. */
    private suspend fun <T> quotaAware(call: suspend () -> T): T = try {
        call()
    } catch (e: CancellationException) {
        throw e
    } catch (e: Throwable) {
        throw e.asQuotaExceededOrNull() ?: e
    }

    override suspend fun deleteCache(cacheId: String) {
        client.async.caches.delete(cacheId, null).await()
    }

    override fun hashContent(contents: List<PromptContent>, settings: PromptSettings?): String {
        val builder = StringBuilder()
        contents.forEach { content ->
            builder.append("role:${content.type}")
            content.parts.forEach { part ->
                builder.append(part.toHashString())
            }
        }
        if (settings is ToolsPromptSettings) {
            // The key is the JSON of the wire `Tool` — exactly what goes into `cachedContents.create`
            // (nested fields, enums, required, wire names). Instruments are sorted so tool order does not matter.
            val tool = GeminiToolDeclarations.toolOf(settings.tools.flatMap { it.instruments }.sortedBy { it.name }, wireKeyLayout)
            if (tool != null) builder.append("tool:").append(tool.toJson())
        }
        return builder.toString().sha256()
    }

    override fun estimateSize(part: PromptPart): Int = when (part) {
        is TextPromptPart -> part.text.length
        // Files are assumed to be large (book / map / pdf) — encourage caching.
        is FileDataPromptPart -> 1_000_000
        is ToolCallPromptPart -> part.call.name.length + part.call.args.size * 10
        is ToolResultPromptPart -> part.name.length + part.result.toString().length
        else -> 0
    }

    private fun PromptPart.toHashString(): String = when (this) {
        is TextPromptPart -> "text:$text"
        is FileDataPromptPart -> "file:$uri"
        is ToolCallPromptPart -> "call:${this.call.name}:" +
            this.call.args.entries.sortedBy { it.key }.joinToString(",") { "${it.key}=${it.value}" }
        is ToolResultPromptPart -> "result:${this.name}:${this.result}"
        else -> toString()
    }
}
