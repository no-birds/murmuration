# Providers

## The executor SPI

A provider is a `PromptExecutor`:

```kotlin
fun interface PromptExecutor {
    fun execute(prompt: Prompt): Flow<PromptExecutionEvent>
    fun supports(llm: Llm): Boolean = true
}
```

The flow emits `PromptExecutionEvent`s (started, parts received, statistics, completed). On any error a `Failed`
event is emitted first and the exception is then propagated through the flow. Provider errors are mapped to the
typed `PromptExecutionException` hierarchy (quota, transient, invalid request, ...), which is what retry
strategies decide on.

`RoutingPromptExecutor(listOf(geminiExecutor, claudeExecutor))` picks the executor by `Llm` through `supports`;
if none accepts the model it fails loudly instead of feeding it to the wrong provider.

Helpers that wrap an executor or a prompt: `withThinkingLevel`, `withIncludeThoughts`, `withMaxOutputTokens`.
`ThinkingLevel` is `MINIMAL`, `LOW`, `MEDIUM` or `HIGH`; each provider maps it to its own mechanism.

## Models and cost

An `Llm` identifies a model: `provider`, `family`, `name`, `version`, `stable`, `model` (the id sent to the API),
an optional `LlmPricingProfile`, supported input and output modalities, and capabilities. Create one with
`Llm(...)`. Each provider module exposes constants and an `ALL` list.

`LlmRegistry` is a global registry. Register your provider's models once at startup, before any pipeline runs:

```kotlin
LlmRegistry.register(GeminiLlms.ALL)
```

`LlmCostCalculator.perModel(pipelineStatistics)` resolves the model name that the API actually returned
(for example a date-suffixed version) with `LlmRegistry.findByActualModel` (longest matching `model` wins) and
prices the usage with the model's profile. Models without a registered profile appear as `null` entries so you
can flag them; cost never silently falls back to zero.

## Gemini (`murmuration-gemini`)

Serves the Gemini API (`provider = "google"`) and Vertex AI (`provider = "vertex"`). Built on
`com.google.genai:google-genai`, which is an `api` dependency.

| Class | Role |
|---|---|
| `GeminiClient` | The `PromptExecutor`. Text, images, embeddings and music through one path. |
| `DirectResponseStrategy`, `StreamingResponseStrategy` | How a response is obtained: one request, or a stream with an idle deadline per chunk. |
| `StreamCallTracker` | Build the SDK client with `.clientOptions(tracker.clientOptions())` and give the same tracker to `StreamingResponseStrategy`; it lets a timed-out stream be torn down for real. |
| `GeminiPromptCachingProvider`, `GeminiCacheConfig` | Gemini context cache, see [caching.md](caching.md). |
| `GeminiLlms` | Model constants: `GeminiLlms.Gemini.*`, `GeminiLlms.Vertex.*`, `GeminiLlms.ALL`, `findModel(provider, model)`. |
| `GeminiRetryStrategy` | HTTP 429 and 5xx, expired cache, and other Gemini-specific failures. |
| `GeminiWireKeyLayout` | Optional tail layout of schema keys, `GeminiWireKeyLayout.NONE` for canonical schemas. |
| `JsonSchema.toGeminiSchema()` | Neutral schema to `com.google.genai.types.Schema`. |

```kotlin
val sdk = Client.builder().vertexAI(true).project("my-project").location("global").build()

val executor = GeminiClient(
    geminiClient = sdk,
    wireKeyLayout = GeminiWireKeyLayout.NONE,
    responseStrategy = DirectResponseStrategy(sdk),
)
LlmRegistry.register(GeminiLlms.ALL)
```

For the Gemini API instead of Vertex AI, build the SDK client without `vertexAI(true)` and use the
`GeminiLlms.Gemini` constants.

Streaming: `DirectResponseStrategy` has an overall timeout (and a per-call OkHttp backstop), so a hung request
becomes a retryable transient error instead of blocking for minutes. `StreamingResponseStrategy(client,
callTracker, idleTimeout)` additionally fails a stream that goes silent for `idleTimeout` with an
`InterruptedIOException`, which `DefaultRetryStrategy` treats as retryable. `HttpOptions.Builder.withoutSdkRetry()`
turns the SDK's own retry layer off, so that Murmuration's retry policy is the only one.

## OpenAI models on Azure (`murmuration-openai`)

Serves OpenAI models hosted on Azure OpenAI (`provider = "azure"`) through the Responses API on the unified
`/openai/v1` endpoint: text with tool calling and structured output, image generation and embeddings.
`com.openai:openai-java` and `azure-identity` are `api` dependencies.

| Class or function | Role |
|---|---|
| `AzureOpenAiClient` | The `PromptExecutor`; maps a canonical model name to the Azure deployment name through `deploymentResolver`. |
| `DirectGptStrategy`, `StreamingGptStrategy` | One request, or a streamed response with idle and overall timeouts. |
| `azureOpenAiClient(endpoint, tokenSupplier, apiVersion)` | Builds the SDK client (unified URL mode, SDK retries off). |
| `azureBearerTokenSupplier(credential, scope)` | Entra token supplier, a `DefaultAzureCredential` by default. |
| `OpenAiLlms` | Model constants (`OpenAiLlms.AZURE_*`) and `OpenAiLlms.ALL`. |
| `OpenAiRetryStrategy` | Retry decisions for OpenAI-specific failures. |

```kotlin
val sdk = azureOpenAiClient(endpoint = "https://my-resource.openai.azure.com", tokenSupplier = azureBearerTokenSupplier())
val executor = AzureOpenAiClient(sdk, DirectGptStrategy(sdk), deploymentResolver = { model -> deployments.getValue(model) })
```

There is no context-cache path: Azure exposes implicit prompt caching only.

## Anthropic (`murmuration-anthropic`)

Serves Claude models through the Messages API. Streaming only.

| Class | Role |
|---|---|
| `AnthropicPromptExecutor` | The `PromptExecutor`; `idleTimeout` and `overallTimeout` are constructor parameters. |
| `AnthropicClientFactory.create(auth, baseUrl)` | Builds the SDK client with SDK retries and SDK logging turned off. |
| `AnthropicAuth.ApiKey`, `AnthropicAuth.Federation` | Authentication is chosen explicitly; environment variables never select it. Secrets are never printed. |
| `AnthropicLlms` | Model specs (`CLAUDE_*`) and `AnthropicLlms.ALL`. |
| `AnthropicRetryStrategy` | Retry decisions for Anthropic-specific failures. |

```kotlin
val client = AnthropicClientFactory.create(AnthropicAuth.ApiKey(System.getenv("ANTHROPIC_API_KEY")))
val executor = AnthropicPromptExecutor(client)
```

Notes: thinking depth comes from the prompt's `thinkingLevel` (thinking cannot be switched off; `MINIMAL` is the
least a model does), a manual thinking budget is ignored, and a prompt's cache strategies are turned into
explicit cache points, see [caching.md](caching.md).

## Writing a provider

1. Implement `PromptExecutor` (and `supports` for routing). Emit `PromptStarted`, parts, usage statistics and a
   final `Completed`; on failure emit `Failed` and throw a `PromptExecutionException` subtype.
2. Expose `Llm` constants with pricing profiles and an `ALL` list.
3. Map `JsonSchema` / `FunctionSchema` to your SDK's types; do not read annotations yourself, the core hands you
   a finished schema.
4. Provide a `RetryStrategy` for the provider's failure modes.
5. Optionally implement `PromptCachingProvider` (or a `CacheBackend` behind the shared `CacheManager`).
