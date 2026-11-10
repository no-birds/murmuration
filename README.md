# Murmuration

[![Maven Central](https://img.shields.io/maven-central/v/org.nobirds.murmuration/murmuration)](https://central.sonatype.com/artifact/org.nobirds.murmuration/murmuration)
[![javadoc](https://javadoc.io/badge2/org.nobirds.murmuration/murmuration/javadoc.svg)](https://javadoc.io/doc/org.nobirds.murmuration/murmuration)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Murmuration is a Kotlin/JVM framework for building LLM agents that call tools. You describe a tool as a plain
Kotlin class, wrap it in an agent, and the framework generates the JSON schema, runs the tool-calling loop,
feeds results back to the model, retries transient failures and keeps track of calls and cost.

The core (`murmuration`) knows nothing about specific LLM vendors. A provider is a separate module that
implements one small interface, `PromptExecutor`; swapping a provider does not touch agent code.

## Why

Agent code in real applications tends to turn into string-concatenated prompts, vendor SDK calls smeared over
business logic, hand-rolled retry and caching wrappers, and tool-calling loops with an implicit state machine.
Murmuration names those pieces and gives them stable contracts:

- `Agent<In, Out>`: a unit you can compose, decorate, retry and observe.
- `@Instrument`: turns a Kotlin function into a model-callable tool; schemas come from the signature.
- `PromptExecutor`: the provider SPI; everything above it is provider-neutral.
- Prompt parts carry cache hints (`PromptCacheStrategy`); providers decide how to honour them.
- Retry policies, call budgets, a model registry with pricing and a cost calculator.

## Installation

Artifacts are published to Maven Central under the group `org.nobirds.murmuration`.
The badge above shows the latest version.

Gradle (Kotlin DSL):

```kotlin
dependencies {
    implementation("org.nobirds.murmuration:murmuration:<version>")
    // pick the provider(s) you use:
    implementation("org.nobirds.murmuration:murmuration-gemini:<version>")
    implementation("org.nobirds.murmuration:murmuration-openai:<version>")
    implementation("org.nobirds.murmuration:murmuration-anthropic:<version>")
}
```

Maven:

```xml
<dependency>
  <groupId>org.nobirds.murmuration</groupId>
  <artifactId>murmuration</artifactId>
  <version><!-- see the Maven Central badge --></version>
</dependency>
```

## Quickstart

A tool is a plain class, an agent is one `toolAgent` call. The code below is compiled and tested in this
repository (`murmuration-samples`), so it cannot drift from the API.

<!-- quickstart:begin -->
```kotlin
import org.nobirds.murmuration.agent.Agent
import org.nobirds.murmuration.llm.Llm
import org.nobirds.murmuration.prompt.PromptExecutor
import org.nobirds.murmuration.prompt.text
import org.nobirds.murmuration.prompt.user
import org.nobirds.murmuration.tooling.Instrument
import org.nobirds.murmuration.tooling.InstrumentParam
import org.nobirds.murmuration.tooling.toolAgent

/** A tool is a plain class: every `@Instrument` method becomes a function the model can call. */
class Calculator {
    var lastSum: Int? = null

    @Instrument("Add two integers and return the sum.")
    fun add(
        @InstrumentParam("First addend") a: Int,
        @InstrumentParam("Second addend") b: Int,
    ): Int = (a + b).also { lastSum = it }
}

/**
 * Builds an agent that lets the model use [Calculator]. The provider is just a [PromptExecutor]:
 * pass `GeminiPromptExecutor`, an OpenAI or an Anthropic executor, or a fake one in tests.
 */
fun calculatorAgent(llm: Llm, executor: PromptExecutor): Agent<String, Int?> =
    toolAgent(
        llm = llm,
        executor = executor,
        tool = Calculator(),
        transform = { _, calculator -> calculator.lastSum },
    ) { _, question, _ ->
        user { text(question) }
    }
```
<!-- quickstart:end -->

To run it, pass an `Llm` constant and an executor of your provider, then run the agent inside a pipeline scope:

```kotlin
val agent = calculatorAgent(llm, executor)
val sum = executePipeline { execute(agent, "What is 2 + 3?") }
```

See [docs/providers.md](docs/providers.md) for how to build an executor for each provider.

## Modules

| Module | Artifact | What it is |
|---|---|---|
| Core | `murmuration` | Agents, prompts, tool calling, schemas, retries, cache manager, model registry. Provider-neutral. |
| Gemini | `murmuration-gemini` | Google Gemini API and Vertex AI: streaming and direct responses, context cache, images. |
| OpenAI | `murmuration-openai` | OpenAI models served by Azure OpenAI, through the Responses API. |
| Anthropic | `murmuration-anthropic` | Claude models through the Messages API, with explicit cache points. |

`murmuration-samples` in this repository holds the compiled examples; it is not published.

## Documentation

- [Agents](docs/agents.md): agents, pipelines, state machines, retries, call budgets, listeners.
- [Tools and schemas](docs/tools-and-schemas.md): `@Instrument`, validation, JSON schemas, structured output.
- [Providers](docs/providers.md): the executor SPI, the three provider modules, model registry and cost.
- [Caching](docs/caching.md): cache strategies, the cache manager, per-provider behaviour.
- [Configuration](docs/configuration.md): every knob the framework exposes and where it lives.
- API reference: [javadoc.io](https://javadoc.io/doc/org.nobirds.murmuration/murmuration).

## Tool error protocol

The framework has to decide whether a tool result is an error (it drives the forced error-visibility round-trip of tool agents and task closing). There is exactly one place that decides: `ErrorMatcher.resolveDefault()`.

- By default a result is an error when its text starts with `ERROR` (case-insensitive, leading whitespace
  ignored).
- To use another convention, implement `org.nobirds.murmuration.agent.ErrorMatcher` and register it in
  `META-INF/services/org.nobirds.murmuration.agent.ErrorMatcher` (Java `ServiceLoader`).
- Zero registered implementations: the default is used. One: it is used. More than one: an
  `IllegalStateException` on the first resolution.
- **Only the final application may register an implementation.** A library built on top of Murmuration must
  never ship that service file: two such files on one classpath make resolution fail. The resolved matcher
  is cached for the JVM, so tests must not swap it either.
- Call `ErrorMatcher.resolveDefault()` once at startup if you want a misconfiguration to fail early rather than
  in the middle of a request.

## Compatibility

Consumers need at least:

```
JVM: 17
Kotlin: 2.3
Kotlin standard library: 2.3.21
```

The modules are compiled with a JDK 21 toolchain against the JVM 17 API and tested on JVM 17 and 21.

## Releases

Release notes are generated by GitHub for every release:
[Releases](https://github.com/no-birds/murmuration/releases). The project is pre-1.0: the public API may change
between minor versions.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md). Report vulnerabilities as described in [SECURITY.md](SECURITY.md).

## License

Apache License 2.0, see [LICENSE](LICENSE).
