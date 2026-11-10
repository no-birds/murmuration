package org.nobirds.murmuration.anthropic

import com.anthropic.client.AnthropicClient
import com.anthropic.client.okhttp.AnthropicOkHttpClient
import com.anthropic.config.AuthenticationConfig
import com.anthropic.config.AuthenticationType
import com.anthropic.config.IdentityTokenConfig
import com.anthropic.config.InMemoryProfileConfigProvider
import com.anthropic.config.ProfileConfig
import com.anthropic.core.LogLevel

/** Name of the header selecting the workspace for API-key requests. */
const val WORKSPACE_HEADER = "anthropic-workspace-id"

/**
 * Builds the SDK client for an explicitly chosen [AnthropicAuth].
 *
 * The builder is never seeded from the environment ([AnthropicOkHttpClient.fromEnv] is not used),
 * so `ANTHROPIC_*` variables cannot select or override the way of signing in. The SDK's own retry
 * layer is off (`maxRetries(0)`): murmuration owns retry and backoff above the provider boundary,
 * and a hidden second layer would double-count attempts and fight `Retry-After` handling. Timeouts
 * stay at the SDK defaults, like the other providers' clients.
 *
 * SDK logging is forced off (`logLevel(OFF)`): the SDK reads `ANTHROPIC_LOG` in the builder's constructor even without
 * `fromEnv()`, and at `debug` it prints full request and response bodies (game prompts) to stderr.
 */
object AnthropicClientFactory {

    /** [baseUrl] null means the SDK default (`https://api.anthropic.com`). */
    fun create(auth: AnthropicAuth, baseUrl: String? = null): AnthropicClient {
        val builder = AnthropicOkHttpClient.builder().maxRetries(0).logLevel(LogLevel.OFF)
        if (baseUrl != null) builder.baseUrl(baseUrl)
        when (auth) {
            is AnthropicAuth.ApiKey -> {
                builder.apiKey(auth.key)
                auth.workspaceId?.let { builder.putHeader(WORKSPACE_HEADER, it) }
            }
            is AnthropicAuth.Federation ->
                builder.configurationProvider(InMemoryProfileConfigProvider.of(federationProfile(auth)))
        }
        return builder.build()
    }

    private fun federationProfile(auth: AnthropicAuth.Federation): ProfileConfig =
        ProfileConfig.builder()
            .authentication(
                AuthenticationConfig.builder()
                    .type(AuthenticationType.OIDC_FEDERATION)
                    .federationRuleId(auth.federationRuleId)
                    .serviceAccountId(auth.serviceAccountId)
                    .identityToken(IdentityTokenConfig.builder().source("file").path(auth.identityTokenPath).build())
                    .build(),
            )
            .organizationId(auth.organizationId)
            .workspaceId(auth.workspaceId)
            .build()
}
