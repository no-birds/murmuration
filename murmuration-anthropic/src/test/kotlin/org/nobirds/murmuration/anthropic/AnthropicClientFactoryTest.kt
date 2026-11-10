package org.nobirds.murmuration.anthropic

import com.anthropic.client.AnthropicClient
import com.anthropic.models.messages.MessageCreateParams
import com.github.tomakehurst.wiremock.WireMockServer
import com.github.tomakehurst.wiremock.client.WireMock.aResponse
import com.github.tomakehurst.wiremock.client.WireMock.absent
import com.github.tomakehurst.wiremock.client.WireMock.equalTo
import com.github.tomakehurst.wiremock.client.WireMock.equalToJson
import com.github.tomakehurst.wiremock.client.WireMock.post
import com.github.tomakehurst.wiremock.client.WireMock.postRequestedFor
import com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo
import com.github.tomakehurst.wiremock.core.WireMockConfiguration.wireMockConfig
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Canonical sample for testing the Anthropic client against WireMock. `build.gradle.kts` puts poison
 * `ANTHROPIC_*` values into the test environment, so every test also proves the environment
 * does not leak into the client.
 */
class AnthropicClientFactoryTest {

    private lateinit var wireMock: WireMockServer

    @TempDir
    lateinit var tmp: Path

    @BeforeEach
    fun setUp() {
        wireMock = WireMockServer(wireMockConfig().dynamicPort())
        wireMock.start()
        wireMock.stubFor(
            post(urlEqualTo("/v1/messages")).willReturn(
                aResponse().withHeader("Content-Type", "application/json").withBody(MESSAGE_JSON),
            ),
        )
    }

    @AfterEach
    fun tearDown() {
        wireMock.stop()
    }

    private fun send(client: AnthropicClient) {
        client.messages().create(
            MessageCreateParams.builder().model("claude-haiku-5-5").maxTokens(16).addUserMessage("hi").build(),
        )
    }

    private fun stubExchange(accessToken: String, expiresIn: Long) {
        wireMock.stubFor(
            post(urlEqualTo("/v1/oauth/token")).willReturn(
                aResponse().withHeader("Content-Type", "application/json")
                    .withBody("""{"access_token":"$accessToken","expires_in":$expiresIn,"token_type":"Bearer"}"""),
            ),
        )
    }

    private fun federation(jwtFile: Path, workspaceId: String? = "wrkspc_w1") = AnthropicAuth.Federation(
        federationRuleId = "fdrl_r1",
        organizationId = "org-1",
        serviceAccountId = "svac_s1",
        workspaceId = workspaceId,
        identityTokenPath = jwtFile.toString(),
    )

    /** The environment carries `ANTHROPIC_LOG=debug` (build.gradle.kts): the SDK would print full bodies to stderr. */
    private fun stderrDuring(block: () -> Unit): String {
        val original = System.err
        val captured = ByteArrayOutputStream()
        System.setErr(PrintStream(captured, true))
        try {
            block()
        } finally {
            System.setErr(original)
        }
        return captured.toString()
    }

    @Test
    fun `ANTHROPIC_LOG in the environment does not make the client print bodies or token exchange to stderr`() {
        val jwt = tmp.resolve("jwt").also { Files.writeString(it, "jwt-secret-1") }
        stubExchange("access-secret-1", expiresIn = 3600)
        val err = stderrDuring {
            send(AnthropicClientFactory.create(AnthropicAuth.ApiKey("k-1"), wireMock.baseUrl()))
            send(AnthropicClientFactory.create(federation(jwt), wireMock.baseUrl()))
        }
        assertEquals("", err)
    }

    @Test
    fun `api key with workspace sends both headers on every request`() {
        val client = AnthropicClientFactory.create(AnthropicAuth.ApiKey("k-1", "wrkspc_w1"), wireMock.baseUrl())
        send(client)
        send(client)
        wireMock.verify(
            2,
            postRequestedFor(urlEqualTo("/v1/messages"))
                .withHeader("x-api-key", equalTo("k-1"))
                .withHeader(WORKSPACE_HEADER, equalTo("wrkspc_w1"))
                .withHeader("Authorization", absent()),
        )
    }

    @Test
    fun `api key without workspace sends no workspace header and ignores env`() {
        send(AnthropicClientFactory.create(AnthropicAuth.ApiKey("k-1"), wireMock.baseUrl()))
        wireMock.verify(
            1,
            postRequestedFor(urlEqualTo("/v1/messages"))
                .withHeader("x-api-key", equalTo("k-1"))
                .withHeader(WORKSPACE_HEADER, absent())
                .withHeader("Authorization", absent()),
        )
    }

    @Test
    fun `federation exchanges the file jwt and sends the access token as bearer`() {
        val jwt = tmp.resolve("jwt").also { Files.writeString(it, "jwt-1\n") }
        stubExchange("access-1", expiresIn = 3600)
        send(AnthropicClientFactory.create(federation(jwt), wireMock.baseUrl()))
        wireMock.verify(
            1,
            postRequestedFor(urlEqualTo("/v1/oauth/token")).withRequestBody(
                equalToJson(
                    """{"grant_type":"urn:ietf:params:oauth:grant-type:jwt-bearer","assertion":"jwt-1",
                    |"federation_rule_id":"fdrl_r1","organization_id":"org-1","service_account_id":"svac_s1",
                    |"workspace_id":"wrkspc_w1"}""".trimMargin(),
                ),
            ),
        )
        wireMock.verify(
            1,
            postRequestedFor(urlEqualTo("/v1/messages"))
                .withHeader("Authorization", equalTo("Bearer access-1"))
                .withHeader("x-api-key", absent())
                .withHeader(WORKSPACE_HEADER, absent()),
        )
    }

    @Test
    fun `federation re-reads the jwt file on every exchange (F7)`() {
        val jwt = tmp.resolve("jwt").also { Files.writeString(it, "jwt-old") }
        // expires_in below the SDK's mandatory-refresh threshold: every request needs a fresh exchange
        stubExchange("access-short", expiresIn = 1)
        val client = AnthropicClientFactory.create(federation(jwt), wireMock.baseUrl())
        send(client)
        Files.writeString(jwt, "jwt-new")
        send(client)
        val assertions = wireMock.findAll(postRequestedFor(urlEqualTo("/v1/oauth/token"))).map { it.bodyAsString }
        assertEquals(2, assertions.size)
        assertTrue("\"assertion\":\"jwt-old\"" in assertions[0].replace(" ", ""), assertions[0])
        assertTrue("\"assertion\":\"jwt-new\"" in assertions[1].replace(" ", ""), assertions[1])
    }

    @Test
    fun `federation caches the access token while it is fresh`() {
        val jwt = tmp.resolve("jwt").also { Files.writeString(it, "jwt-1") }
        stubExchange("access-1", expiresIn = 3600)
        val client = AnthropicClientFactory.create(federation(jwt), wireMock.baseUrl())
        send(client)
        send(client)
        wireMock.verify(1, postRequestedFor(urlEqualTo("/v1/oauth/token")))
        wireMock.verify(2, postRequestedFor(urlEqualTo("/v1/messages")))
    }

    @Test
    fun `federation without workspace omits workspace_id from the exchange`() {
        val jwt = tmp.resolve("jwt").also { Files.writeString(it, "jwt-1") }
        stubExchange("access-1", expiresIn = 3600)
        send(AnthropicClientFactory.create(federation(jwt, workspaceId = null), wireMock.baseUrl()))
        val body = wireMock.findAll(postRequestedFor(urlEqualTo("/v1/oauth/token"))).single().bodyAsString
        assertFalse("workspace_id" in body, body)
    }

    @Test
    fun `fromConfig treats empty strings as not set and picks exactly one way`() {
        val key = AnthropicAuth.fromConfig(apiKey = "k", workspaceId = "", federationRuleId = "", organizationId = "")
        assertTrue(key is AnthropicAuth.ApiKey && key.workspaceId == null)
        val fed = AnthropicAuth.fromConfig(
            apiKey = "", federationRuleId = "fdrl_r1", organizationId = "org-1",
            serviceAccountId = "svac_s1", workspaceId = "wrkspc_w1", identityTokenPath = "/var/run/jwt",
        )
        assertTrue(fed is AnthropicAuth.Federation && fed.workspaceId == "wrkspc_w1")
        assertFailsWith<IllegalArgumentException> { AnthropicAuth.fromConfig() }
        assertFailsWith<IllegalArgumentException> { AnthropicAuth.fromConfig(apiKey = "k", federationRuleId = "fdrl_r1") }
        assertFailsWith<IllegalArgumentException> { AnthropicAuth.fromConfig(federationRuleId = "fdrl_r1") }
    }

    @Test
    fun `secrets are not printed`() {
        assertFalse("secret-key" in AnthropicAuth.ApiKey("secret-key", "wrkspc_w1").toString())
        val e = assertFailsWith<IllegalArgumentException> { AnthropicAuth.fromConfig(apiKey = "secret-key", identityTokenPath = "/p") }
        assertFalse("secret-key" in e.message.orEmpty())
    }

    private companion object {
        const val MESSAGE_JSON = """{"id":"msg_1","type":"message","role":"assistant","model":"claude-haiku-5-5",
            "content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","stop_sequence":null,
            "usage":{"input_tokens":1,"output_tokens":1}}"""
    }
}
