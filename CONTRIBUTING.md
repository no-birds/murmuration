# Contributing

Thanks for your interest in Murmuration. This page explains how to build and test the project, how releases
work, and the few rules that apply to every change.

## Build and test

Requirements: a JDK 21 (the Gradle toolchain compiles with it; the published bytecode targets JVM 17) and
nothing else, the Gradle wrapper is in the repository.

```
./gradlew build                      # compile, test and check every module
./gradlew :murmuration:test --rerun  # one module's tests, forced to run again
scripts/check-docs.sh                # the documentation checks CI runs
```

- Always add `--rerun` right after a test task when you want a real run: without it Gradle may report an
  up-to-date or cached result without executing a single test. Check the `tests=` count in the XML reports under
  `build/test-results/` if in doubt.
- Test JVM: CI runs the tests on JVM 17, the declared minimum. Compilation uses the JDK 21 toolchain. Select the test
  JVM locally with `-Pmurmuration.testJvm=17` (or `21`); both JDKs must be installed.
- Tests must not call a real LLM or any paid provider; use fake executors (see `murmuration-samples`).
- The declared minimum for consumers (JVM 17, Kotlin 2.3) lives in one place, the `kotlin` convention plugin in
  `build-logic`. `./gradlew printCompatibility` prints it and the README's Compatibility section must match.
  Raising it is a breaking change and needs a deliberate pull request.
- The quickstart in the README is the `// region quickstart` block of `murmuration-samples`, byte for byte.
  Change the sample, then paste the region into the README.

## Releases

Versions come from git tags (`v<major>.<minor>.<patch>`) through the axion release plugin. No file holds a version,
and documentation must not name a Murmuration version number (use the Maven Central badge).

- A push to `main` publishes a **patch** release.
- A **minor** or **major** release is started manually with the release workflow's `workflow_dispatch`, choosing the
  increment.
- Every release runs the full CI first (including the documentation and language checks). The publish job then
  waits for approval of the protected environment before anything is uploaded to Maven Central. The release
  workflow also creates the `v<version>` tag and a GitHub Release with generated release notes; there is no
  changelog file.
- Contributors do not release. Maintainers approve the environment.

## Rules

- **Everything is in English**: code, comments, KDoc, documentation, commit messages, issues and pull requests.
  CI fails if any Cyrillic character is found in the tree.
- **Link only to issues of this repository.** No references to external trackers or private systems.
- **Never upload artifacts to the Central Portal (or any repository) by hand, and never publish outside the release
  workflow.** The workflow is the only path; it is what makes releases reproducible and verified.
- Do not store secrets, credentials, internal hostnames or personal paths in the repository.
- Public API changes need KDoc, and the affected page in `docs/` if behaviour users depend on changes.
- Keep pull requests focused and make sure `./gradlew build` and `scripts/check-docs.sh` pass.

## Reporting problems

Bugs and questions: open an issue in this repository. Vulnerabilities: see [SECURITY.md](SECURITY.md), not a
public issue.
