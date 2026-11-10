package org.nobirds.murmuration.agent

/**
 * Base exception for agent-side failures (program-guard, call-limit, framework-internal
 * inconsistencies). Provider-side prompt execution errors are represented by a separate hierarchy,
 * [org.nobirds.murmuration.prompt.PromptExecutionException], typed by error kind;
 * retry/recovery logic looks at types, not at flags.
 */
open class AgentExecutionException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

/**
 * Subclass for problems at the prompt-executor level that are not related to the provider API
 * (for example, a wrong result type from the executor). Provider-side errors go to
 * [org.nobirds.murmuration.prompt.PromptExecutionException].
 */
open class PromptExecutorException(message: String, cause: Throwable? = null) : AgentExecutionException(message, cause)

class PipelineLimitException(message: String) : RuntimeException(message)
