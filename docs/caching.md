# Caching

Murmuration models caching as an intent on the prompt, and leaves the mechanism to the provider.

## Marking prompt parts

Every prompt part carries a `PromptCacheStrategy`:

| Strategy | Meaning |
|---|---|
| `PromptCacheStrategy.NO_CACHE` | Default for parts you create. Never cached. |
| `PromptCacheStrategy.PIPELINE` | Stable for the lifetime of a pipeline run; cached, and released when the run ends. |
| `PromptCacheStrategy.ttl(duration)` | Stable for at least `duration`, independent of the pipeline. |

```kotlin
user {
    text(instructions, cacheStrategy = PromptCacheStrategy.PIPELINE)   // stable prefix
    text(userInput)                                                    // fresh, not cached
}
```

Put stable content first (system prompt, instructions, schemas) and volatile content (snapshots, user input)
after it. `inputSchema` and `jsonWithSchema` already cache the schema part for the pipeline and leave the value
uncached.

## Who places the cache

`Llm.cacheKind` is derived from the model's provider, never configured:

| `CacheKind` | Providers | What happens |
|---|---|---|
| `MARKS` | Anthropic | The request itself carries cache breakpoints (`cache_control`). The tool-calling agent marks the history it creates with a sliding tail mark (`CachePlacement.SLIDING_TAIL`). |
| `EXECUTOR` | Gemini, Azure OpenAI | The executor decides. The agent places no marks (`CachePlacement.NONE`). |

## Gemini: context cache

`GeminiPromptCachingProvider(client, wireKeyLayout, storage, config)` is a `PromptCachingProvider` that binds the
shared `CacheManager` to a Gemini backend (`GeminiCacheBackend`). Pass it to `GeminiClient(cachingProvider = ...)`.

Lifecycle:

1. The cacheable parts of the prompt are collected and hashed; the hash is looked up in your
   `PromptContextCacheStorage`.
2. On a miss, a Gemini cache is created, and the entry is stored with its expiration.
3. On a hit, the cached parts are cut out of the request and the cache is referenced instead. If the entry is
   close to expiring (`refreshDuration`), its TTL is extended before use.
4. While a stream is in flight the cache is kept alive in the background (`cacheKeepAliveInterval` of
   `GeminiClient`, 60 s by default).
5. Cache creation and usage are reported into the pipeline statistics (cache-create rows, cached token counts,
   TTL and scope), so cost reports can price cache reads, writes and storage.

You implement `PromptContextCacheStorage` (find by hash or id, save, update expiration, delete) on top of your own
database; the framework assumes nothing about the store. Entries are `PromptContextCacheEntry` values.

`PromptCachingProvider.createPipelineListener(scope, emitter)` returns a listener that deletes `PIPELINE`-scoped
caches when the pipeline finishes and then reports `onCacheDeleted`, so the real cache lifetime can be recorded
for storage billing. Registering that listener on the pipeline is up to the application.

### Configuration

`GeminiCacheConfig` (a thin layer over `CacheManagerConfig`):

| Field | Default | Meaning |
|---|---|---|
| `ttl` | 10 min | Base TTL of created caches; a `ttl(...)` strategy on a part can lower it. |
| `refreshDuration` | 5 min | Extend a cache before use when less than this remains. |
| `minCacheSizeBytes` | 4096 | Smallest increment that justifies creating a new, longer cache. |
| `reorderStrategy` | `GroupByCacheability` | Which parts go into the cache. |
| `growthStrategy` | `CacheGrowthStrategy.AlwaysGrow` | Whether to grow a longer cache on top of a found prefix. |

`reorderStrategy`: `Strict` takes the contiguous cacheable prefix and stops at the first `NO_CACHE` part
(the order seen by the model is preserved; a warning is logged when cacheable parts were skipped).
`GroupByCacheability` collects every cacheable part, across `NO_CACHE` holes, into the cache and sends the fresh
parts as the tail. This changes the order the model sees (cached first), and it is the default because stable
schemas and instructions often sit after volatile blocks inside one message.

`growthStrategy`: `AlwaysGrow` extends the cache whenever the new remainder is above `minCacheSizeBytes`. Each
extension is a full-price write, and some are never read (for example after the last round-trip). Implement
`CacheGrowthStrategy` for a different policy.

### Other providers through the shared manager

`CacheManager(backend, storage, config)` is provider-neutral. A provider that wants the same lifecycle implements
`CacheBackend` (create, refresh and delete a cache, hash content, estimate the size of a part) and gets lookup,
refresh, growth and pipeline cleanup for free.

## Anthropic: cache points

The Anthropic executor turns strategies into explicit cache points with one rule, in request order (tools, system,
messages):

- a point stands only at the end of the continuous stable prefix; the first `NO_CACHE` part ends it, and no point
  is put after a hole;
- at most 4 points per request, parts are never reordered;
- one TTL class for all points: 1 hour if any covered part wishes longer than 5 minutes, otherwise 5 minutes;
- nothing is placed unless at least one part of the prefix was marked cacheable by its author (tools alone are
  not a reason to pay for a write).

A prefix shorter than the model's minimum cacheable length is not an error: the API skips the point.

## Azure OpenAI

Only the service's implicit prompt caching applies. The executor emits no cache-create events; keep stable content
at the start of the prompt to benefit from it.
