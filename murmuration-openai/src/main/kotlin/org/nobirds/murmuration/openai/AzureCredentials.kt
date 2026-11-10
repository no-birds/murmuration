package org.nobirds.murmuration.openai

import com.azure.core.credential.TokenCredential
import com.azure.core.credential.TokenRequestContext
import com.azure.identity.DefaultAzureCredentialBuilder
import com.openai.azure.AzureOpenAIServiceVersion
import com.openai.azure.AzureUrlPathMode
import com.openai.client.OpenAIClient
import com.openai.client.okhttp.OpenAIOkHttpClient
import com.openai.credential.BearerTokenCredential
import java.util.function.Supplier

/**
 * Returns a supplier of Azure Entra bearer tokens for the Cognitive Services scope. The
 * [credential] (a `DefaultAzureCredential` by default: env → workload identity → managed
 * identity → Azure CLI, etc.) caches and refreshes tokens internally, so the supplier is safe
 * to call on every request — [OpenAIOkHttpClient] does exactly that through
 * [BearerTokenCredential] to pick up a fresh token after a 401/AUTH_REFRESH retry.
 */
fun azureBearerTokenSupplier(
    credential: TokenCredential = DefaultAzureCredentialBuilder().build(),
    scope: String = "https://cognitiveservices.azure.com/.default",
): () -> String = {
    credential.getTokenSync(TokenRequestContext().addScopes(scope)).token
}

/**
 * Builds an Azure-flavoured [OpenAIClient] for the given [endpoint] (the bare resource URL,
 * e.g. `https://my-res.openai.azure.com`; the `/openai/v1` suffix is appended here if missing,
 * so configs keep the bare URL). The SDK runs in **UNIFIED** Azure URL mode against the
 * next-generation `/openai/v1` base path: LEGACY mode (per-deployment
 * `/openai/deployments/{d}/...?api-version=...` URLs) was the right choice for Chat
 * Completions, but Azure serves the **Responses API only on the unified `/openai/v1` path** —
 * LEGACY-style `/responses` requests come back 404. UNIFIED mode appends no path segments of
 * its own (only LEGACY does), hence the suffix is baked into the base URL; the `model(...)` in
 * request params still must be the Azure **deployment name** (resolved by
 * [AzureOpenAiClient]'s `deploymentResolver`).
 *
 * [apiVersion] is the Azure `api-version` query param; blank → `preview` (the unified path
 * uses `preview`/`latest` version aliases — legacy dated versions like `2025-03-01-preview`
 * are rejected there with "API version not supported").
 *
 * The SDK's built-in retry layer is disabled ([OpenAIOkHttpClient.Builder.maxRetries] = 0):
 * murmuration owns retry/backoff above the provider boundary, and a hidden second layer would
 * double-count attempts and fight our `Retry-After` handling.
 */
fun azureOpenAiClient(
    endpoint: String,
    tokenSupplier: () -> String,
    apiVersion: String = "",
): OpenAIClient =
    OpenAIOkHttpClient.builder()
        .baseUrl(endpoint.trimEnd('/').removeSuffix("/openai/v1") + "/openai/v1")
        .credential(BearerTokenCredential.create(Supplier { tokenSupplier() }))
        .azureUrlPathMode(AzureUrlPathMode.UNIFIED)
        .azureServiceVersion(AzureOpenAIServiceVersion.fromString(apiVersion.ifBlank { "preview" }))
        .maxRetries(0)
        .build()
