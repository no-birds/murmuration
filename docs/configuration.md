# Configuration

Murmuration has no configuration files, environment variables or global switches of its own. Everything is
configured in code, at the place where the object is constructed. This page lists the knobs and where they live.

## At startup

| What | How |
|---|---|
| Model registry | `LlmRegistry.register(Provider.ALL)` once, before any pipeline. |
| Tool error protocol | Optional `META-INF/services/org.nobirds.murmuration.agent.ErrorMatcher` in the final application; call `ErrorMatcher.resolveDefault()` once to fail fast. See the [README](../README.md#tool-error-protocol). |
| Provider client | Built by you from the provider SDK, then wrapped (`GeminiClient`, `AzureOpenAiClient`, `AnthropicPromptExecutor`). See [providers.md](providers.md). |

Provider modules never read credentials from the environment on their own: Anthropic authentication is an explicit
`AnthropicAuth`, Azure authentication is an explicit token supplier, and the Gemini SDK client is built by you.
The Azure OpenAI and Anthropic client builders turn off the SDK's own retries (the Anthropic one also its SDK
logging), and `withoutSdkRetry()` does the same for the Gemini SDK, so that the retry policy below is the only
retry layer.

## Per agent

| Setting | Where |
|---|---|
| Model | `llm` parameter of the factory (`textAgent`, `toolAgent`). |
| Provider | the `executor` parameter. |
| Turn limit | `maxTurns` / `maxTurnsFn` of `toolAgent`. |
| Per-round-trip retry | `roundTripRetry = RetryPolicy(...)` of `toolAgent`; off by default. |
| Whole-run retry | `agent.withRetry(...)` or `RetryAgent(agent, RetryPolicy(...))`. |
| Recovery | `agent.withRecovery<E>(recoveryAgent)`. |
| Cache of framework instructions | `frameworkInstructionsCacheStrategy` of `toolAgent` (default `PIPELINE`). |
| Listener | `eventListener` of `toolAgent`. |
| Tool decorators | `InstrumentFactory(DecoratorRegistry(...))`; validators and transformers through `ValidatorRegistry` and `TransformerRegistry`. |

## Per prompt

`PromptBuilder` settings: the `llm`, `settings` (`PromptSettings`: thinking level, thought visibility, maximum
output tokens, stream idle timeout; typed-output and tool settings are set by the agents) and per-part cache
strategies. See [caching.md](caching.md).

## Per pipeline

| Setting | Where |
|---|---|
| Call budget | `executePipeline(maxCalls, callLimitStrategy)` or `executePipelineWithBudget(budgetStrategy)`. |
| Listeners | `pipelineEventListener` parameter; compose with `CompositePipelineEventListener`. |
| Statistics | `PipelineStatisticsCollector` as a listener; `LlmCostCalculator.perModel(stats)` for cost. |
| Pipeline id | `pipelineId` parameter (random by default). |

## Retry policy

`RetryPolicy(strategy, maxRetries = 3, initialBackoff = 2 s, maxBackoff = 10 min, quotaBackoffs = [15 s, 30 s, 60 s],
quotaJitter = 0.2)`. Quota errors (429 and the like) and all other errors have separate attempt budgets; without a
server hint a quota error waits along `quotaBackoffs` (the last value is the ceiling) with jitter; a server
retry-after hint is honoured up to 120 s.

## Caching

`CacheManagerConfig` / `GeminiCacheConfig`, see [caching.md](caching.md#configuration). Anthropic and Azure have no
cache configuration: Anthropic cache points follow the part strategies, Azure caches implicitly.

## Logging

Murmuration logs through SLF4J (`slf4j-api` is an `api` dependency). Add the binding of your choice; the framework
never configures logging.
