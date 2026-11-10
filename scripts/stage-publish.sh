#!/usr/bin/env bash
# Usage: stage-publish.sh <version> <sha> <staging-dir>
#
# Publishes the 4 modules into a local staging repository, signed with an EPHEMERAL GPG key generated
# here (no secrets involved), then checks what landed there:
#   - every artifact file has a signature that verifies against the ephemeral public key;
#   - POM metadata (tools/check-poms.py);
#   - V-9: the build SHA is recorded in scm.tag of every POM;
#   - V-7 (stage mode): the Central upload task is not registered.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
# shellcheck source=lib.sh
source "$here/lib.sh"

version="${1:?usage: stage-publish.sh <version> <sha> <staging-dir>}"
sha="${2:?usage: stage-publish.sh <version> <sha> <staging-dir>}"
staging="${3:?usage: stage-publish.sh <version> <sha> <staging-dir>}"
require_release_semver "$version"
rm -rf "$staging"
mkdir -p "$staging"

export GNUPGHOME
GNUPGHOME="$(mktemp -d)"
trap 'rm -rf "$GNUPGHOME"' EXIT
chmod 700 "$GNUPGHOME"

gpg --batch --quiet --pinentry-mode loopback --passphrase '' \
  --quick-generate-key 'murmuration stage <stage@invalid.example>' rsa3072 sign 1d
key_file="$GNUPGHOME/stage-secret.asc"
gpg --batch --quiet --pinentry-mode loopback --passphrase '' --armor \
  --export-secret-keys 'stage@invalid.example' >"$key_file"

# GITHUB_ACTIONS=false: staging must not register the Central upload (V-7, stage mode).
# The same variable is what switches registration on in the convention plugin.
export ORG_GRADLE_PROJECT_signingInMemoryKey
ORG_GRADLE_PROJECT_signingInMemoryKey="$(cat "$key_file")"
export ORG_GRADLE_PROJECT_signingInMemoryKeyPassword=""

cd "$root"
GITHUB_ACTIONS=false ./gradlew publishAllPublicationsToTestRepository \
  "-Prelease.forceVersion=$version" "-Pmurmuration.testRepo=$staging" "-Pmurmuration.buildSha=$sha" --console=plain

echo "--- V-7 (stage mode): no Central upload task registered"
if GITHUB_ACTIONS=false ./gradlew tasks --all --console=plain -q | grep -q 'publishToMavenCentral'; then
  echo "error: publishToMavenCentral is registered in the stage build" >&2
  exit 1
fi

echo "--- signatures"
count=0
while read -r path; do
  case "$path" in
    *.asc) continue ;;
  esac
  [ -f "$staging/$path" ] || { echo "error: missing $path in staging" >&2; exit 1; }
  [ -f "$staging/$path.asc" ] || { echo "error: missing signature for $path" >&2; exit 1; }
  gpg --batch --quiet --verify "$staging/$path.asc" "$staging/$path"
  count=$((count + 1))
done < <(artifact_files "$version")
echo "verified $count signatures"

echo "--- POM metadata"
python3 -I "$root/tools/check-poms.py" "$staging" "$sha"

echo "--- V-9: build SHA in scm.tag of every POM"
for a in "${ARTIFACTS[@]}"; do
  pom="$staging/$GROUP_PATH/$a/$version/$a-$version.pom"
  grep -q "<tag>$sha</tag>" "$pom" || { echo "error: scm.tag in $pom is not $sha" >&2; exit 1; }
done
echo "SHA $sha present in all POMs"
