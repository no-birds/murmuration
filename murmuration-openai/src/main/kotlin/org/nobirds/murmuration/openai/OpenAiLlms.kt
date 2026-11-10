package org.nobirds.murmuration.openai

import org.nobirds.murmuration.openai.OpenAiLlms.ALL
import org.nobirds.murmuration.openai.OpenAiLlms.findModel
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.llm.LlmCostCalculator
import org.nobirds.murmuration.llm.LlmPricingProfile
import org.nobirds.murmuration.llm.LlmRegistry
import org.nobirds.murmuration.llm.llm

/**
 * [Llm] constants for Azure OpenAI (GPT-5.4 family + gpt-image-2) with their pricing
 * profiles attached. Mirror of `GeminiLlms`: pricing lives directly on the model
 * ([Llm.pricingProfile]); [LlmCostCalculator] resolves a model by its actual API name
 * through [LlmRegistry], to which [ALL] is registered at application startup.
 *
 * All models use `provider = "azure"` and `family = "gpt"`. The requested `model` string
 * is the canonical model name (`gpt-5.4`, `gpt-image-2`, …); the actual Azure **deployment
 * name** is resolved separately by [AzureOpenAiClient]'s `deploymentResolver`, so it is not
 * baked into these constants.
 *
 * The [findModel] lookup is kept for application-side wiring (selecting a model from
 * configuration).
 */
object OpenAiLlms {

    // GPT-5.4 (<272k context, Global Standard): input $2.50/1M, cached input $0.25/1M,
    // output $15/1M. No explicit prompt cache create/storage on Azure chat completions
    // (implicit prompt caching only) → storage rate stays 0. Source: Azure OpenAI Service
    // pricing page (azure.microsoft.com/en-us/pricing/details/azure-openai/), checked 2026-07-04.
    private val GPT_5_4_PRICING = LlmPricingProfile(
        inputPricePerMillion = 2.50,
        cachedInputPricePerMillion = 0.25,
        outputPricePerMillion = 15.0,
        cacheStoragePricePerMillionPerHour = 0.0,
    )

    // GPT-5.4-mini (Global Standard): input $0.75/1M, cached $0.075/1M, output $4.50/1M.
    // Source: Azure OpenAI Service pricing page, checked 2026-07-04.
    private val GPT_5_4_MINI_PRICING = LlmPricingProfile(
        inputPricePerMillion = 0.75,
        cachedInputPricePerMillion = 0.075,
        outputPricePerMillion = 4.50,
        cacheStoragePricePerMillionPerHour = 0.0,
    )

    // GPT-5.4-nano (Global Standard): input $0.20/1M, cached $0.02/1M, output $1.25/1M.
    // Source: Azure OpenAI Service pricing page, checked 2026-07-04.
    private val GPT_5_4_NANO_PRICING = LlmPricingProfile(
        inputPricePerMillion = 0.20,
        cachedInputPricePerMillion = 0.02,
        outputPricePerMillion = 1.25,
        cacheStoragePricePerMillionPerHour = 0.0,
    )

    // GPT-5.6 Terra (deployed 2026-07-10, model version 2026-07-09, GlobalStandard): pricing per
    // the 09.07 announcement — same tier as gpt-5.4 ($2.50/$0.25/$15). Azure's pricing page had
    // not yet published a dedicated 5.6 entry as of this deployment (az/portal expose no billing
    // metadata) — verify against the first actual invoice.
    private val GPT_5_6_TERRA_PRICING = LlmPricingProfile(
        inputPricePerMillion = 2.50,
        cachedInputPricePerMillion = 0.25,
        outputPricePerMillion = 15.0,
        cacheStoragePricePerMillionPerHour = 0.0,
    )

    // GPT-5.6 Luna (deployed 2026-07-10, model version 2026-07-09, GlobalStandard): pricing per
    // the 09.07 announcement — +33% over gpt-5.4-mini ($1.00/$0.10/$6). Azure's pricing page had
    // not yet published a dedicated 5.6 entry as of this deployment — verify against the first
    // actual invoice.
    private val GPT_5_6_LUNA_PRICING = LlmPricingProfile(
        inputPricePerMillion = 1.00,
        cachedInputPricePerMillion = 0.10,
        outputPricePerMillion = 6.0,
        cacheStoragePricePerMillionPerHour = 0.0,
    )

    // GPT-5 mini (Aug 2025): input $0.25/1M, cached $0.025/1M, output $2.00/1M. No longer the
    // interim fallback — gpt-5.4-nano was deployed 2026-07-04 (GlobalStandard, 500K TPM quota)
    // and is the live implementor model; this constant is retained in the catalogue as an
    // additional selectable older-tier model.
    private val GPT_5_MINI_PRICING = LlmPricingProfile(
        inputPricePerMillion = 0.25,
        cachedInputPricePerMillion = 0.025,
        outputPricePerMillion = 2.00,
        cacheStoragePricePerMillionPerHour = 0.0,
    )

    // gpt-image-2: text input $5/1M, cached $1.25/1M, image (output) tokens $30/1M.
    // Approximation: image-input tokens are billed at a separate higher rate ($8/1M) that we
    // do NOT model as its own field — our image prompts are text-only, so the text-input rate
    // is what applies. Refine when an actual Azure invoice lands.
    private val GPT_IMAGE_2_PRICING = LlmPricingProfile(
        inputPricePerMillion = 5.0,
        cachedInputPricePerMillion = 1.25,
        outputPricePerMillion = 30.0,
        cacheStoragePricePerMillionPerHour = 0.0,
    )

    // text-embedding-3-large: input-only $0.13/1M tokens. Embeddings have no output tokens,
    // no caching, no storage tier — only the input rate matters for cost reporting. Source:
    // Azure OpenAI Service pricing page (azure.microsoft.com/en-us/pricing/details/azure-openai/),
    // checked 2026-07-10.
    private val TEXT_EMBEDDING_3_LARGE_PRICING = LlmPricingProfile(
        inputPricePerMillion = 0.13,
        cachedInputPricePerMillion = 0.0,
        outputPricePerMillion = 0.0,
        cacheStoragePricePerMillionPerHour = 0.0,
    )

    // text-embedding-3-large is natively 3072-d, but we always request the reduced 768-d vector
    // (via the Embeddings API `dimensions` param) to match the pgvector `vector(768)` columns the
    // application stores into — so the model declares 768 as the dimensionality it produces here.
    private const val TEXT_EMBEDDING_3_LARGE_DIMENSIONS = 768

    // GPT-5.4 text/chat models share one capability shape: text + image input, text output
    // (NO thoughts — Azure chat completions bill reasoning tokens but never return the CoT
    // trace, so declaring `output { thoughts() }` would be a lie), level-based thinking and
    // tool calling. No `caching(...)` capability: Azure chat exposes only implicit prompt
    // caching, there is no explicit `cache.create` to floor. Only name / model / pricing vary.
    private fun gptTextModel(
        displayName: String,
        model: String,
        pricingProfile: LlmPricingProfile,
        version: String = "5.4",
    ): Llm = llm("azure", model) {
        family = "gpt"
        name = displayName
        this.version = version
        stable = true
        pricing = pricingProfile
        input { text(); image() }
        output { text() }
        thinking { level() }
        toolCalling()
    }

    val AZURE_GPT_5_4 = gptTextModel("GPT-5.4", "gpt-5.4", GPT_5_4_PRICING)

    val AZURE_GPT_5_4_MINI = gptTextModel("GPT-5.4 mini", "gpt-5.4-mini", GPT_5_4_MINI_PRICING)

    val AZURE_GPT_5_4_NANO = gptTextModel("GPT-5.4 nano", "gpt-5.4-nano", GPT_5_4_NANO_PRICING)

    // GPT-5.6 family (announced/deployed 09.07-10.07): same capability shape as gpt-5.4 (text +
    // thinking level + tool calling, no explicit cache floor). System card notes increased
    // agentic drive versus 5.4 — that affects prompting/gates elsewhere, not this registration.
    // Sol (the top tier of the family) is intentionally NOT deployed/registered here — too
    // expensive per todo; only Terra and Luna are live.
    val AZURE_GPT_5_6_TERRA =
        gptTextModel("GPT-5.6 Terra", "gpt-5.6-terra", GPT_5_6_TERRA_PRICING, version = "5.6")

    val AZURE_GPT_5_6_LUNA =
        gptTextModel("GPT-5.6 Luna", "gpt-5.6-luna", GPT_5_6_LUNA_PRICING, version = "5.6")

    // Older mini tier; same capability shape as the 5.4 family (text+image in, text out,
    // reasoning_effort, tools). `version` on the constant reflects the actual family.
    val AZURE_GPT_5_MINI = llm("azure", "gpt-5-mini") {
        family = "gpt"; name = "GPT-5 mini"; version = "5"; stable = true
        pricing = GPT_5_MINI_PRICING
        input { text(); image() }
        output { text() }
        thinking { level() }
        toolCalling()
    }

    // Image model: text+image in (images.edit accepts reference images), image out (Images
    // API). No thinking / tools / caching.
    val AZURE_GPT_IMAGE_2 = llm("azure", "gpt-image-2") {
        family = "gpt"; name = "gpt-image-2"; version = "2"; stable = true
        pricing = GPT_IMAGE_2_PRICING
        input { text(); image() }
        output { image() }
    }

    // Text embedding model — used by the embedding path (AiServiceImpl.embed via config), not
    // for generation. text + embedding-output; no thinking / tools / caching.
    val AZURE_TEXT_EMBEDDING_3_LARGE = llm("azure", "text-embedding-3-large") {
        family = "embedding"; name = "Text Embedding 3 Large"; version = "3"; stable = true
        pricing = TEXT_EMBEDDING_3_LARGE_PRICING
        input { text() }
        output { embedding(TEXT_EMBEDDING_3_LARGE_DIMENSIONS) }
    }

    /**
     * All known Azure OpenAI [Llm] constants.
     *
     * `by lazy` mirrors `GeminiLlms.ALL`: it avoids a class-loader init-order hazard when a
     * caller touches a nested constant before the enclosing object is fully initialised.
     */
    val ALL: List<Llm> by lazy {
        listOf(
            AZURE_GPT_5_4, AZURE_GPT_5_4_MINI, AZURE_GPT_5_4_NANO, AZURE_GPT_5_MINI,
            AZURE_GPT_5_6_TERRA, AZURE_GPT_5_6_LUNA,
            AZURE_GPT_IMAGE_2, AZURE_TEXT_EMBEDDING_3_LARGE,
        )
    }

    /**
     * Resolves an [Llm] constant by provider + actual model name. `matches` is a prefix
     * match, so `gpt-5.4` also prefix-matches `gpt-5.4-nano`; the longest-[model] wins to
     * pick the most specific constant (same semantics as `GeminiLlms.findModel`).
     */
    fun findModel(provider: String, model: String): Llm =
        ALL.filter { it.provider == provider && it.matches(model) }
            .maxByOrNull { it.model.length }
            ?: throw IllegalArgumentException("LLM model $model not found")
}
