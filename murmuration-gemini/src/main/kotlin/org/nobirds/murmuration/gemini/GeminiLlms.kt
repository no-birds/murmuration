package org.nobirds.murmuration.gemini

import org.nobirds.murmuration.gemini.GeminiLlms.ALL
import org.nobirds.murmuration.gemini.GeminiLlms.findModel
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.llm.LlmCostCalculator
import org.nobirds.murmuration.llm.LlmPricingProfile
import org.nobirds.murmuration.llm.LlmRegistry
import org.nobirds.murmuration.llm.llm
import org.nobirds.murmuration.prompt.ThinkingLevel

/**
 * [Llm] constants for the Google Gemini API and Vertex AI with their pricing profiles
 * attached. Pricing lives directly on the model (see [Llm.pricingProfile]);
 * [LlmCostCalculator] resolves a model by its actual API name through [LlmRegistry], to
 * which [ALL] is registered at application startup.
 *
 * The [findModel] lookup is kept for application-side wiring (selecting a model from
 * configuration) — the actual config-to-model functions live with the application code,
 * we don't drag application configuration into this module.
 */
object GeminiLlms {

    // Gemini 3 Flash Preview: text/image/video input $0.5/1M, cached input $0.05/1M,
    // text output $3/1M, context cache storage $1/1M/hour.
    private val GEMINI_3_FLASH_PRICING = LlmPricingProfile(
        inputPricePerMillion = 0.5,
        cachedInputPricePerMillion = 0.05,
        outputPricePerMillion = 3.0,
        cacheStoragePricePerMillionPerHour = 1.0,
    )

    // Gemini 3.1 Flash-Lite (Global tier): text/image/video input $0.25/1M, cached
    // $0.025/1M, text output $1.5/1M. Cache storage taken from the Flash sibling profile
    // ($1/1M/h) — Google hasn't published a separate storage rate for Lite at the time
    // of writing; refine if data appears. Non-global tier rates are 10% higher across
    // the board (we charge global by default since we run in the global region).
    private val GEMINI_3_1_FLASH_LITE_PRICING = LlmPricingProfile(
        inputPricePerMillion = 0.25,
        cachedInputPricePerMillion = 0.025,
        outputPricePerMillion = 1.5,
        cacheStoragePricePerMillionPerHour = 1.0,
    )

    // Gemini 3.5 Flash-Lite: text/image/video input $0.30/1M, cached $0.03/1M,
    // text output $2.50/1M, context cache storage $1.0/1M/h.
    private val GEMINI_3_5_FLASH_LITE_PRICING = LlmPricingProfile(
        inputPricePerMillion = 0.30,
        cachedInputPricePerMillion = 0.03,
        outputPricePerMillion = 2.50,
        cacheStoragePricePerMillionPerHour = 1.0,
    )

    // Gemini 3.5 Flash: text/image/video/audio input $1.5/1M, cached $0.15/1M, output
    // $9/1M, cache storage ~$1/1M/h. Audio input rate matches text — no separate
    // field in [LlmPricingProfile] is needed.
    private val GEMINI_3_5_FLASH_PRICING = LlmPricingProfile(
        inputPricePerMillion = 1.5,
        cachedInputPricePerMillion = 0.15,
        outputPricePerMillion = 9.0,
        cacheStoragePricePerMillionPerHour = 1.0,
    )

    private val GEMINI_3_6_FLASH_PRICING = LlmPricingProfile(
        inputPricePerMillion = 1.5,
        cachedInputPricePerMillion = 0.15,
        outputPricePerMillion = 7.50,
        cacheStoragePricePerMillionPerHour = 1.0,
    )

    // Gemini 3.7 Flash (released 2026-08-13): introductory pricing through 2026-12-31 —
    // input $0.75/1M, output $3.75/1M (50% off the standard $1.5/$7.5 that applies from
    // 2027-01-01; REVISIT this profile then). Cached input assumed at 10% of input
    // ($0.075/1M) and cache storage at $1/1M/h per the family pattern — refine when
    // Google publishes the exact rates.
    private val GEMINI_3_7_FLASH_PRICING = LlmPricingProfile(
        inputPricePerMillion = 0.75,
        cachedInputPricePerMillion = 0.075,
        outputPricePerMillion = 3.75,
        cacheStoragePricePerMillionPerHour = 1.0,
    )

    // Gemini 3.8 Flash: per the Google launch blog the price equals 3.7 Flash — introductory
    // $0.75/1M input, $3.75/1M output through 2026-12-31, standard $1.5/$7.5 from 2027-01-01
    // (REVISIT this profile then). Cached input assumed at 10% of input ($0.075/1M) and cache
    // storage at $1/1M/h per the family pattern — refine when Google publishes the exact rates.
    private val GEMINI_3_8_FLASH_PRICING = LlmPricingProfile(
        inputPricePerMillion = 0.75,
        cachedInputPricePerMillion = 0.075,
        outputPricePerMillion = 3.75,
        cacheStoragePricePerMillionPerHour = 1.0,
    )

    // Gemini 3.1 Flash Image Preview: input (text, image) $0.5/1M, image output $60/1M,
    // thinking $3/1M (a separate rate — for an image model thinking is NOT billed at the image price!). Previously
    // thinking was billed at $60/1M (image output) and overstated the image cost ~20x.
    // Caching is not used for the image model.
    private val GEMINI_3_1_FLASH_IMAGE_PRICING = LlmPricingProfile(
        inputPricePerMillion = 0.5,
        cachedInputPricePerMillion = 0.0,
        outputPricePerMillion = 60.0,
        cacheStoragePricePerMillionPerHour = 0.0,
        thinkingPricePerMillion = 3.0,
    )

    // Gemini 3.1 Flash-Lite Image: cheap image tier for background/token art. Rates derived
    // from Vertex AI pricing (checked 2026-07-15): a 1K image is billed at $0.034, and image
    // output is metered at 1120 tokens per 1K image → $0.034 / 1120 tok ≈ $30/1M output
    // tokens (half the Flash-Image sibling's $60/1M). Input (text, image) $0.25/1M — mirrors
    // the Flash-Lite text tier and is half the Flash-Image $0.5/1M. Thinking $1.50/1M (own
    // tariff — image-model thinking is NOT billed at the image rate, same as the sibling).
    // Caching is not used for the image model.
    private val GEMINI_3_1_FLASH_LITE_IMAGE_PRICING = LlmPricingProfile(
        inputPricePerMillion = 0.25,
        cachedInputPricePerMillion = 0.0,
        outputPricePerMillion = 30.0,
        cacheStoragePricePerMillionPerHour = 0.0,
        thinkingPricePerMillion = 1.5,
    )

    // Gemini 3.1 Pro Preview pricing for ≤200K input tokens.
    // The cost calculator is flat (no tiered profile) — for prompts > 200K the actual cost
    // is 2× input and 1.5× output. Our typical turn stays under 200K, so the underestimate
    // is rare and bounded. If we start regularly hitting tier 2, switch the calculator to a
    // tiered profile rather than baking the higher rate here.
    private val GEMINI_3_1_PRO_PRICING = LlmPricingProfile(
        inputPricePerMillion = 2.0,
        cachedInputPricePerMillion = 0.2,
        outputPricePerMillion = 12.0,
        cacheStoragePricePerMillionPerHour = 4.5,
    )

    // Gemini 2.5 Pro pricing for ≤200K input tokens (stable production tier).
    // For prompts >200K the actual cost is 2× input ($2.50/M) and 1.5× output ($15/M); we
    // keep the flat ≤200K rate here for the same reason as the 3.1 Pro profile — typical
    // turns stay under the threshold, switch to a tiered calculator only if regularly
    // exceeded.
    private val GEMINI_2_5_PRO_PRICING = LlmPricingProfile(
        inputPricePerMillion = 1.25,
        cachedInputPricePerMillion = 0.31,
        outputPricePerMillion = 10.0,
        cacheStoragePricePerMillionPerHour = 4.5,
    )

    // gemini-embedding-001: input-only $0.15/1M (batch $0.075). Embeddings have no output
    // tokens, no caching, no storage tier — only the input rate matters for cost reporting.
    private val GEMINI_EMBEDDING_001_PRICING = LlmPricingProfile(
        inputPricePerMillion = 0.15,
        cachedInputPricePerMillion = 0.0,
        outputPricePerMillion = 0.0,
        cacheStoragePricePerMillionPerHour = 0.0,
    )

    // Native embedding dimensionality of gemini-embedding-001 — declared on the model
    // (Modality.Embedding) and matches the pgvector column the application stores into.
    private const val EMBEDDING_001_DIMENSIONS = 768

    // Lyria 3 Pro is billed per generation (a flat per-track price), not per token — ~$0.08/track
    // (approximate: Vertex Lyria 3 Pro pricing isn't publicly published, anchored on Lyria 2's
    // $0.06/30s). Token fields stay zero; cost = perRequestUsd × generations (see LlmCostCalculator).
    private val LYRIA_3_PRO_PRICING = LlmPricingProfile(
        inputPricePerMillion = 0.0,
        cachedInputPricePerMillion = 0.0,
        outputPricePerMillion = 0.0,
        perRequestUsd = 0.08,
    )

    // Gemini text/chat models share one capability shape: multimodal input, text + thoughts
    // output, prompt caching (provider token floor differs by family), level-based thinking and
    // tool calling. Only name / model / pricing / cache floor vary — collapsed into this builder so
    // the capability shape lives in one place. `cacheTokens`: the `cache.create` minimum (Caching).
    private fun geminiTextModel(
        provider: String,
        displayName: String,
        modelVersion: String,
        isStable: Boolean,
        model: String,
        pricingProfile: LlmPricingProfile,
        cacheTokens: Int,
        thinkingLevels: Array<ThinkingLevel> = emptyArray(),
    ): Llm = llm(provider, model) {
        family = "gemini"
        name = displayName
        version = modelVersion
        stable = isStable
        pricing = pricingProfile
        input { text(); image(); audio(); video() }
        output { text(); thoughts() }
        caching(minTokens = cacheTokens)
        // Empty `thinkingLevels` = no declared set (pass-through); non-empty = the supported tiers.
        thinking { level(*thinkingLevels) }
        toolCalling()
    }

    object Gemini {

        val GEMINI_3_FLASH = geminiTextModel(
            "google",
            "Gemini 3.1 Flash",
            "3.1",
            false,
            "gemini-3-flash-preview",
            GEMINI_3_FLASH_PRICING,
            4096
        )

        val GEMINI_3_5_FLASH = geminiTextModel(
            "google",
            "Gemini 3.5 Flash",
            "3.5",
            true,
            "gemini-3.5-flash",
            GEMINI_3_5_FLASH_PRICING,
            4096
        )

        val GEMINI_3_6_FLASH = geminiTextModel(
            "google",
            "Gemini 3.6 Flash",
            "3.6",
            false,
            "gemini-3.6-flash",
            GEMINI_3_6_FLASH_PRICING,
            4096
        )

        val GEMINI_3_7_FLASH = geminiTextModel(
            "google",
            "Gemini 3.7 Flash",
            "3.7",
            false,
            "gemini-3.7-flash",
            GEMINI_3_7_FLASH_PRICING,
            4096
        )

        val GEMINI_3_8_FLASH = geminiTextModel(
            "google",
            "Gemini 3.8 Flash",
            "3.8",
            false,
            "gemini-3.8-flash",
            GEMINI_3_8_FLASH_PRICING,
            4096,
            // 3.8 declares its tiers: the config-level check (ai.models.*.thinkingLevel) validates against them.
            arrayOf(ThinkingLevel.LOW, ThinkingLevel.MEDIUM, ThinkingLevel.HIGH),
        )

        val GEMINI_3_1_FLASH_LITE = geminiTextModel(
            "google",
            "Gemini 3.1 Flash-Lite",
            "3.1",
            true,
            "gemini-3.1-flash-lite",
            GEMINI_3_1_FLASH_LITE_PRICING,
            4096
        )

        val GEMINI_3_5_FLASH_LITE = geminiTextModel(
            "google",
            "Gemini 3.5 Flash-Lite",
            "3.5",
            true,
            "gemini-3.5-flash-lite",
            GEMINI_3_5_FLASH_LITE_PRICING,
            4096
        )

        val GEMINI_3_1_PRO = geminiTextModel(
            "google",
            "Gemini 3.1 Pro",
            "3.1",
            false,
            "gemini-3.1-pro-preview",
            GEMINI_3_1_PRO_PRICING,
            4096
        )

        val GEMINI_2_5_PRO =
            geminiTextModel("google", "Gemini 2.5 Pro", "2.5", true, "gemini-2.5-pro", GEMINI_2_5_PRO_PRICING, 1024)

        // Image model: text+image in, text+image out (generateContent multimodal). No caching/tools.
        val GEMINI_3_1_FLASH_IMAGE = llm("google", "gemini-3.1-flash-image") {
            family = "gemini"; name = "Gemini 3.1 Flash Image"; version = "3.1"; stable = true
            pricing = GEMINI_3_1_FLASH_IMAGE_PRICING
            input { text(); image() }
            output { text(); image() }
        }

        // Cheap image model for background/token art. Same capability shape as the sibling,
        // plus a declared thinking-tier constraint: live probes (2026-07-15) show LOW is
        // rejected with a 400 ("Thinking level LOW is not supported"), while MINIMAL and HIGH
        // both work. Without this declaration `clampThinkingLevel` is a no-op and the default
        // LOW would 400 in production.
        val GEMINI_3_1_FLASH_LITE_IMAGE = llm("google", "gemini-3.1-flash-lite-image") {
            family = "gemini"; name = "Gemini 3.1 Flash-Lite Image"; version = "3.1"; stable = true
            pricing = GEMINI_3_1_FLASH_LITE_IMAGE_PRICING
            input { text(); image() }
            output { text(); image() }
            thinking { level(ThinkingLevel.MINIMAL, ThinkingLevel.HIGH) }
        }

        internal val ALL = listOf(
            GEMINI_3_FLASH,
            GEMINI_3_5_FLASH,
            GEMINI_3_6_FLASH,
            GEMINI_3_7_FLASH,
            GEMINI_3_8_FLASH,
            GEMINI_3_1_FLASH_LITE,
            GEMINI_3_5_FLASH_LITE,
            GEMINI_3_1_PRO,
            GEMINI_2_5_PRO,
            GEMINI_3_1_FLASH_IMAGE,
            GEMINI_3_1_FLASH_LITE_IMAGE
        )
    }

    object Vertex {

        val VERTEX_GEMINI_3_FLASH = geminiTextModel(
            "vertex",
            "Gemini 3 Flash",
            "3",
            false,
            "gemini-3-flash-preview",
            GEMINI_3_FLASH_PRICING,
            4096
        )

        val VERTEX_GEMINI_3_5_FLASH = geminiTextModel(
            "vertex",
            "Gemini 3.5 Flash",
            "3.5",
            true,
            "gemini-3.5-flash",
            GEMINI_3_5_FLASH_PRICING,
            4096
        )

        val VERTEX_GEMINI_3_6_FLASH = geminiTextModel(
            "vertex",
            "Gemini 3.6 Flash",
            "3.6",
            false,
            // The Vertex catalog publishes 3.6 without "-preview" (the direct-API variant has it)
            "gemini-3.6-flash",
            GEMINI_3_6_FLASH_PRICING,
            4096
        )

        val VERTEX_GEMINI_3_7_FLASH = geminiTextModel(
            "vertex",
            "Gemini 3.7 Flash",
            "3.7",
            false,
            // Vertex catalog publishes it without the "-preview" suffix (direct API — with it),
            // same as the 3.6 sibling.
            "gemini-3.7-flash",
            GEMINI_3_7_FLASH_PRICING,
            4096
        )

        val VERTEX_GEMINI_3_8_FLASH = geminiTextModel(
            "vertex",
            "Gemini 3.8 Flash",
            "3.8",
            false,
            // Vertex catalog publishes it without the "-preview" suffix (direct API — with it),
            // same as the 3.6/3.7 siblings.
            "gemini-3.8-flash",
            GEMINI_3_8_FLASH_PRICING,
            4096,
            // 3.8 declares its tiers: the config-level check (ai.models.*.thinkingLevel) validates against them.
            arrayOf(ThinkingLevel.LOW, ThinkingLevel.MEDIUM, ThinkingLevel.HIGH),
        )

        val VERTEX_GEMINI_3_1_FLASH_LITE = geminiTextModel(
            "vertex",
            "Gemini 3.1 Flash-Lite",
            "3.1",
            true,
            "gemini-3.1-flash-lite",
            GEMINI_3_1_FLASH_LITE_PRICING,
            4096
        )

        val VERTEX_GEMINI_3_5_FLASH_LITE = geminiTextModel(
            "vertex",
            "Gemini 3.5 Flash-Lite",
            "3.5",
            true,
            "gemini-3.5-flash-lite",
            GEMINI_3_5_FLASH_LITE_PRICING,
            4096
        )

        val VERTEX_GEMINI_3_1_PRO = geminiTextModel(
            "vertex",
            "Gemini 3.1 Pro",
            "3.1",
            false,
            "gemini-3.1-pro-preview",
            GEMINI_3_1_PRO_PRICING,
            4096
        )

        val VERTEX_GEMINI_2_5_PRO =
            geminiTextModel("vertex", "Gemini 2.5 Pro", "2.5", true, "gemini-2.5-pro", GEMINI_2_5_PRO_PRICING, 1024)

        // Image model: text+image in, text+image out (generateContent multimodal). No caching/tools.
        val VERTEX_GEMINI_3_1_FLASH_IMAGE = llm("vertex", "gemini-3.1-flash-image") {
            family = "gemini"; name = "gemini 3.1 Flash Image"; version = "3.1"; stable = true
            pricing = GEMINI_3_1_FLASH_IMAGE_PRICING
            input { text(); image() }
            output { text(); image() }
        }

        // Cheap image model for background/token art. Same capability shape as the sibling,
        // plus a declared thinking-tier constraint: live probes (2026-07-15) show LOW is
        // rejected with a 400 ("Thinking level LOW is not supported"), while MINIMAL and HIGH
        // both work. Without this declaration `clampThinkingLevel` is a no-op and the default
        // LOW would 400 in production.
        val VERTEX_GEMINI_3_1_FLASH_LITE_IMAGE = llm("vertex", "gemini-3.1-flash-lite-image") {
            family = "gemini"; name = "gemini 3.1 Flash-Lite Image"; version = "3.1"; stable = true
            pricing = GEMINI_3_1_FLASH_LITE_IMAGE_PRICING
            input { text(); image() }
            output { text(); image() }
            thinking { level(ThinkingLevel.MINIMAL, ThinkingLevel.HIGH) }
        }

        // Lyria 3 Pro: text prompt in, audio out (generateContent with AUDIO response modality).
        val VERTEX_LYRIA_3_PRO = llm("vertex", "lyria-3-pro-preview") {
            family = "lyria"; name = "Lyria 3 Pro"; version = "3"
            pricing = LYRIA_3_PRO_PRICING
            input { text() }
            output { audio() }
        }

        // Text embedding model — used by GeminiEmbeddingExecutor, not for generation.
        val VERTEX_GEMINI_EMBEDDING_001 = llm("vertex", "gemini-embedding-001") {
            family = "embedding"; name = "Gemini Embedding 001"; version = "001"; stable = true
            pricing = GEMINI_EMBEDDING_001_PRICING
            input { text() }
            output { embedding(EMBEDDING_001_DIMENSIONS) }
        }

        internal val ALL = listOf(
            VERTEX_GEMINI_3_FLASH,
            VERTEX_GEMINI_3_5_FLASH,
            VERTEX_GEMINI_3_6_FLASH,
            VERTEX_GEMINI_3_7_FLASH,
            VERTEX_GEMINI_3_8_FLASH,
            VERTEX_GEMINI_3_1_FLASH_LITE,
            VERTEX_GEMINI_3_5_FLASH_LITE,
            VERTEX_GEMINI_3_1_PRO,
            VERTEX_GEMINI_2_5_PRO,
            VERTEX_GEMINI_3_1_FLASH_IMAGE,
            VERTEX_GEMINI_3_1_FLASH_LITE_IMAGE,
            VERTEX_LYRIA_3_PRO,
            VERTEX_GEMINI_EMBEDDING_001
        )
    }

    /**
     * All known [Llm] constants for Google Gemini API and Vertex AI.
     *
     * `by lazy` is essential here: an eager `Gemini.ALL + Vertex.ALL` initialisation
     * deadlocks the JVM class loader when callers touch `GeminiLlms.Vertex.X` first —
     * loading `Vertex` triggers the enclosing `GeminiLlms.<clinit>`, which itself reads
     * `Vertex.ALL` while `Vertex.<clinit>` is still running, yielding a null and an NPE.
     */
    val ALL: List<Llm> by lazy { Gemini.ALL + Vertex.ALL }

    fun findModel(provider: String, model: String): Llm {
        val cleanModel = model.removePrefix("models/")
        return ALL.filter {
            it.provider == provider && (
                it.matches(model) ||
                it.matches(cleanModel) ||
                it.model.removePrefix("models/").removeSuffix("-preview") == cleanModel.removeSuffix("-preview")
            )
        }
            .maxByOrNull { it.model.length }
            ?: throw IllegalArgumentException("LLM model $model not found")
    }
}
