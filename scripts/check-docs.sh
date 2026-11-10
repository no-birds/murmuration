#!/usr/bin/env bash
# Mechanical documentation checks (the `docs` step of ci.yml). Needs a Gradle-capable environment
# for checks 3 and 6. Reports every failure, then exits non-zero if there was any.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
cd "$root"
fail=0
bad() { echo "docs: FAIL: $*"; fail=1; }

required=(README.md LICENSE CONTRIBUTING.md SECURITY.md
  docs/agents.md docs/tools-and-schemas.md docs/providers.md docs/caching.md docs/configuration.md)
doc_files=(README.md CONTRIBUTING.md SECURITY.md docs)

echo "== 1. required files exist"
for f in "${required[@]}"; do [ -f "$f" ] || bad "missing $f"; done

echo "== 2. no murmuration version numbers in docs"
if grep -rnE 'org\.nobirds\.murmuration:[a-z-]+:[0-9]|\b0\.1\.0\b' "${doc_files[@]}" 2>/dev/null; then
  bad "hardcoded version found (use the Maven Central badge, not a number)"
fi

echo "== 3. murmuration-samples compiles on the declared minimum"
./gradlew :murmuration-samples:compileKotlin --console=plain -q || bad "murmuration-samples does not compile"

echo "== 4. README quickstart equals the sample region byte for byte"
sample="$(grep -rl '// region quickstart' murmuration-samples/src 2>/dev/null | head -n 1)"
if [ -z "$sample" ]; then
  bad "no '// region quickstart' in murmuration-samples"
else
  awk '/\/\/ region quickstart/{f=1;next} /\/\/ endregion/{f=0} f' "$sample" >"${RUNNER_TEMP:-/tmp}/quickstart.sample"
  awk '/<!-- quickstart:begin -->/{b=1;next} /<!-- quickstart:end -->/{b=0} b && /^```/{next} b' README.md \
    >"${RUNNER_TEMP:-/tmp}/quickstart.readme"
  [ -s "${RUNNER_TEMP:-/tmp}/quickstart.readme" ] || bad "README has no quickstart block between quickstart markers"
  diff -u "${RUNNER_TEMP:-/tmp}/quickstart.sample" "${RUNNER_TEMP:-/tmp}/quickstart.readme" || bad "quickstart differs from the sample"
fi

echo "== 5. no internal markers in docs"
while IFS= read -r pattern; do
  case "$pattern" in ''|'#'*) continue ;; esac
  if grep -rnE -- "$pattern" "${doc_files[@]}" 2>/dev/null; then bad "internal marker '$pattern'"; fi
done <"$here/internal-markers.txt"

echo "== 6. README Compatibility section equals printCompatibility"
section="$(awk '/^#+ +Compatibility/{f=1;next} /^#+ /{f=0} f' README.md)"
if ! compat="$(./gradlew printCompatibility --console=plain -q)"; then
  bad "printCompatibility failed"
else
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    grep -qxF -- "$line" <<<"$section" || bad "Compatibility section lacks the line: $line"
  done <<<"$compat"
fi

[ "$fail" -eq 0 ] && echo "docs: OK"
exit "$fail"
