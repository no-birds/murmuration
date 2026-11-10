# Agents

An agent is a unit of work with a typed input and output. Everything else (retry, recovery, tool loops,
pipelines) is built from that one interface.

```kotlin
interface Agent<In, Out> {
    val name: String
    suspend fun ExecutionContext.execute(input: In): Out
}
```

## Pipelines and the execution context

Agents run inside a pipeline. A pipeline carries the call budget, the listeners and the statistics.

```kotlin
val result = coroutineScope.executePipeline {
    execute(agent, input)                      // runs and waits
    val deferred = scheduleExecute(agent, input) // runs concurrently, returns Deferred
    deferred.await()
}
```

- `CoroutineScope.executePipeline(...)` builds the run, calls your block with an `ExecutionContext` receiver and
  tears down listeners and pending coroutines when the block ends. Parameters: `pipelineId`,
  `pipelineEventListener`, `maxCalls` and `callLimitStrategy` (see below).
- `CoroutineScope.executePipelineWithBudget(...)` is the same with a `BudgetStrategy` instead of a call count.
- `execute(agent, input)` and `scheduleExecute(agent, input)` are the extensions to call inside the block.

## Building agents

| Factory | Produces |
|---|---|
| `agent { input -> ... }` | A `LambdaAgent`: arbitrary suspend code. |
| `textAgent(executor, serializer) { ctx, input -> ... }` | One LLM call; the prompt is built by the block, the answer is decoded by a `TextPromptSerializer`. |
| `textAgent<In, Out>(llm, executor) { ctx, input -> ... }` | Same, with structured output: `Out` must be `@Serializable`, the JSON schema is generated from it. |
| `toolAgent(...)` / `toolCallingAgent(...)` | A tool-calling loop, see below. |
| `imageAgent`, `embeddingAgent`, `musicAgent` | Media and embedding calls for providers that support them. |

The prompt block receives a `PromptBuilder` receiver (`system { text(...) }`, `user { text(...) }`), see
[tools-and-schemas.md](tools-and-schemas.md) and [caching.md](caching.md) for the parts you can attach to it.

Agents compose with `agent.map { ctx, out -> ... }` and `agent.mapInput { ctx, in -> ... }`.

## Tool-calling agents

`toolAgent` runs the canonical loop: build the prompt, call the model, execute the tool calls it requested,
append the results, repeat until the model answers without tool calls (or `maxTurns` / the budget runs out).

```kotlin
val agent = toolAgent(
    llm = llm,
    executor = executor,
    tool = Calculator(),                       // one instance, lives as long as the agent
    transform = { _, calculator -> calculator.lastSum },
) { _, question, _ -> user { text(question) } }
```

Notes:

- Prefer the `tool = ...` overload. The `factory = ...` overload is deprecated: it creates the tool on every
  `execute`, so tool state is lost when an outer retry re-executes the agent.
- Useful parameters: `maxTurns`, `maxTurnsFn`, `eventListener`, `roundTripRetry`, `name`.
- Tool results are visible to the model only in the next round-trip. Tool calls of one response run in the
  phase the round-trip started in.
- A tool that throws is reported to the model as an `Error: ...` result and the loop goes on. Throw
  `AgentAbortException` to abort the whole run instead.
- Arguments that do not fit the tool's declared parameters are refused with a single message listing every
  problem (`ToolArgumentException`); the model corrects them on the next round-trip.
- An identical repeated call gets an advisory appended to its result; further identical repeats are skipped as
  no-ops. Mark gates and positional mutators with `@NeverSkip` to opt out.
- Whether a result counts as an error is decided by `ErrorMatcher`, see the
  [README](../README.md#tool-error-protocol).

### State machines

A tool object can also be a `Stateful` implementation: the agent then runs in phases. Each phase has its own
instructions, tools are gated by `@Phases("A", "B")` (or `@AllPhases`), and the model moves between phases with
the built-in `transitionTo` tool. `stateMachine(initial) { state(...) { transition(...) }; terminal(...) }`
declares the graph; `defaultStateMachine()` / `defaultStateful()` give the trivial `WORKING -> FINISHED` machine
that every plain tool gets.

Transitions are applied once, at the end of a round-trip, never in the middle of a batch: `transitionTo` only
registers the intent (its gate is checked immediately, so the model gets an honest success or failure), and
`Stateful.onAfterToolCalls` applies it after all calls of the response have run. A `Transition` may carry a
suspend `effect` that runs after the check passes and before the state flips, which makes a transition a unit
of work (for example, running a batch of sub-agents). A failing effect does not kill the run: the transition is
not applied, the reason reaches the model in the next phase snapshot, and the model can request it again.

### Composing tool sets

`buildInstruments { from(toolA) { exclude("x"); inPhases(...) }; from(toolB) { include("y") } }` builds a tool
surface from several donor objects. `AgentToolMount` is the single place that decides what is mounted on an
agent (declarations and phase defaults).

## Retries

There are two layers; both use `RetryPolicy` (strategy, `maxRetries`, `initialBackoff`, `maxBackoff`, quota
backoffs).

- **Per round-trip (inner).** `toolAgent(roundTripRetry = RetryPolicy(...))`, off by default. A retryable failure
  of one round-trip (stream break, transient 5xx or 429, malformed response) discards that attempt and re-prompts
  on the preserved history. It costs one round-trip instead of replaying the whole invocation.
- **Whole invocation (outer).** `agent.withRetry(maxRetries, initialBackoff, maxBackoff, retryStrategy)` or
  `RetryAgent(agent, policy)` re-executes the entire agent. It is the fallback when the inner budget is used up
  or the failure happens outside the tool loop.

`RetryStrategy` decides what is retryable. `DefaultRetryStrategy` covers typed `PromptExecutionException`
subclasses and common I/O and JSON failures; provider modules add `GeminiRetryStrategy`, `OpenAiRetryStrategy`
and `AnthropicRetryStrategy`; combine with `DefaultRetryStrategy or GeminiRetryStrategy`. A server-supplied
retry-after hint is honoured (capped at 120 s). `AgentAbortException` is never retried.

For operations that are not agents (cache creation, side-channel calls) use `retryWithTimeout`.

`RecoveryAgent` / `agent.withRecovery<E>(recoveryAgent)` is different from retry: on a specific
`PromptExecutionException` subtype it runs a recovery agent that transforms the input (for example, rewrites a
prompt rejected by a safety filter) and executes again. `withPromptRecovery` is the variant whose recovery
agent receives the failing prompt.

## Budgets

A pipeline can limit how much an agent run may spend.

- `executePipeline(maxCalls = 50, callLimitStrategy = ...)`: caps LLM round-trips. `DefaultCallLimitStrategy`
  stops gracefully (a tool agent returns a partial result, single-shot agents throw `PipelineLimitException`);
  `ThrowingCallLimitStrategy` always throws.
- `executePipelineWithBudget(budgetStrategy = DefaultBudgetStrategy(BudgetLimits(...)))`: limits on tool calls,
  requests, failures, retries and outer iterations, evaluated on a snapshot of `ExecutionMetrics`. Implement
  `BudgetStrategy` for your own policy.

## Observability

- `PipelineEventListener` receives pipeline- and agent-level events; `pipelineEventListener { ... }` builds one
  from lambdas and `CompositePipelineEventListener` fans out to several.
- `AgentEventListener` is the per-agent marker; `ToolCallingListener` adds tool-calling events (before/after a
  tool call, tool errors, state transitions, exhausted turns or budget).
- `PipelineStatisticsCollector` aggregates calls, tokens and cache usage into `PipelineStatistics`; feed it to
  `LlmCostCalculator.perModel(stats)` to get a cost breakdown per model (see
  [providers.md](providers.md#models-and-cost)).
- `PromptExecutionEvent` is the stream every executor emits (started, parts received, statistics, completed,
  failed).
