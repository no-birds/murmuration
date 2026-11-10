# CLAUDE.md

Murmuration is a Kotlin/JVM framework for LLM agents with tools. The core `:murmuration` is provider-neutral;
provider modules `:murmuration-gemini`, `:murmuration-openai` and `:murmuration-anthropic` implement it.
All modules are published as Maven artifacts under `org.nobirds.murmuration`.

## Build

- Build everything: `./gradlew build`.
- Shared build conventions live in `build-logic` (`org.nobirds.murmuration.library`).

## Release

- Run the `release` workflow (`workflow_dispatch`); it publishes to Maven Central.
- The version comes from git tags via axion. Do not store a version in any file.

## Rules

- Everything is in English: code, comments, commit messages, issues.
- Link only to issues of this repository.
