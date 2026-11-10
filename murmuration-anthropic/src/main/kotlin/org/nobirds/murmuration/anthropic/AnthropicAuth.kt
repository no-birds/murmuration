package org.nobirds.murmuration.anthropic

/**
 * How the Anthropic client authenticates. Exactly one way is chosen explicitly; the process
 * environment (`ANTHROPIC_API_KEY`, `ANTHROPIC_FEDERATION_RULE_ID`, ...) never selects it.
 *
 * Secrets are never part of [toString] or of exception messages.
 */
sealed interface AnthropicAuth {

    /**
     * Static API key (`x-api-key`). [workspaceId] (`wrkspc_...`) is sent as the
     * `anthropic-workspace-id` header on every request when set.
     */
    class ApiKey(val key: String, val workspaceId: String? = null) : AnthropicAuth {
        init {
            require(key.isNotBlank()) { "Anthropic API key must not be blank" }
            require(workspaceId == null || workspaceId.isNotBlank()) { "workspaceId must be null or non-blank" }
        }

        override fun toString(): String = "ApiKey(workspaceId=$workspaceId)"
    }

    /**
     * Workload identity federation: the SDK exchanges the JWT read from [identityTokenPath]
     * (`urn:ietf:params:oauth:grant-type:jwt-bearer` at `/v1/oauth/token`) for an access token and
     * caches it. The file is re-read on every exchange. [workspaceId] scopes the minted token; the
     * `anthropic-workspace-id` header is not used with federation tokens.
     *
     * **Correctness condition (F7).** A projected service-account token is one and the same JWT
     * (one `jti`) until the kubelet rotates it (at ~80% of `expirationSeconds`, at least 600 s), and
     * Anthropic rejects a replayed `jti`. So the access-token lifetime configured on the federation
     * rule must be >= the projected token rotation period; otherwise a re-exchange happens with the
     * same JWT and fails. The token `audience` in the pod manifest must be `https://api.anthropic.com`.
     */
    class Federation(
        val federationRuleId: String,
        val organizationId: String,
        val serviceAccountId: String,
        val workspaceId: String? = null,
        val identityTokenPath: String,
    ) : AnthropicAuth {
        init {
            require(federationRuleId.startsWith("fdrl_")) { "federationRuleId must start with 'fdrl_'" }
            require(organizationId.isNotBlank()) { "organizationId must not be blank" }
            require(serviceAccountId.startsWith("svac_")) { "serviceAccountId must start with 'svac_'" }
            require(workspaceId == null || workspaceId.startsWith("wrkspc_") || workspaceId == "default") {
                "workspaceId must start with 'wrkspc_' or be 'default'"
            }
            require(identityTokenPath.isNotBlank()) { "identityTokenPath must not be blank" }
        }

        override fun toString(): String =
            "Federation(federationRuleId=$federationRuleId, organizationId=$organizationId, " +
                "serviceAccountId=$serviceAccountId, workspaceId=$workspaceId, identityTokenPath=$identityTokenPath)"
    }

    companion object {
        /**
         * Builds the auth from flat config values; an empty string means "not set".
         * Fails unless exactly one way is configured: an API key, or the complete federation set.
         */
        fun fromConfig(
            apiKey: String? = null,
            workspaceId: String? = null,
            federationRuleId: String? = null,
            organizationId: String? = null,
            serviceAccountId: String? = null,
            identityTokenPath: String? = null,
        ): AnthropicAuth {
            val key = apiKey.orNull()
            val workspace = workspaceId.orNull()
            val federationFields = mapOf(
                "federationRuleId" to federationRuleId.orNull(),
                "organizationId" to organizationId.orNull(),
                "serviceAccountId" to serviceAccountId.orNull(),
                "identityTokenPath" to identityTokenPath.orNull(),
            )
            val federationSet = federationFields.values.any { it != null }
            require(!(key != null && federationSet)) {
                "Anthropic auth is ambiguous: both an API key and federation settings are configured"
            }
            if (key != null) return ApiKey(key, workspace)
            require(federationSet) { "Anthropic auth is not configured: set an API key or federation settings" }
            val missing = federationFields.filterValues { it == null }.keys
            require(missing.isEmpty()) { "Anthropic federation settings are incomplete, missing: ${missing.joinToString()}" }
            return Federation(
                federationRuleId = federationFields.getValue("federationRuleId")!!,
                organizationId = federationFields.getValue("organizationId")!!,
                serviceAccountId = federationFields.getValue("serviceAccountId")!!,
                workspaceId = workspace,
                identityTokenPath = federationFields.getValue("identityTokenPath")!!,
            )
        }

        private fun String?.orNull(): String? = this?.takeIf { it.isNotBlank() }
    }
}
