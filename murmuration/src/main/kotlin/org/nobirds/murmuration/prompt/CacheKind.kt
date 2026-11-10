package org.nobirds.murmuration.prompt

/**
 * How a model's provider caches prompt context. Derived from the model's provider (see
 * [org.nobirds.murmuration.llm.cacheKind]), never stored or configured: the kind and the request's
 * addressee come from the same field and cannot diverge.
 */
enum class CacheKind {
    /** The caller marks cache breakpoints inside the request itself (Anthropic `cache_control`). */
    MARKS,

    /** The provider's executor decides (Gemini explicit/implicit caching, Azure prefix caching); the caller places nothing. */
    EXECUTOR,
}
