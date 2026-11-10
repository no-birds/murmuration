#!/usr/bin/env bash
# Usage: tag.sh <version> <sha>
#
# Creates tag v<version> on <sha>, pushes it, and creates a GitHub Release with generated notes.
# Idempotent: tag already on this SHA -> skipped; tag on another SHA -> refusal;
# release already exists -> skipped.
#
# Environment:
#   SIDE_EFFECT_WRAP  command prefix put in front of the two outward side effects (the tag push and
#                     the release creation). Empty in the release job; the stage job sets it to
#                     "echo" so both are printed instead of executed. Nothing else differs.
#   GH_TOKEN          token for `gh` (needs contents: write when the side effects are real).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$here/lib.sh"

version="${1:?usage: tag.sh <version> <sha>}"
sha="${2:?usage: tag.sh <version> <sha>}"
require_release_semver "$version"
tag="v$version"

# shellcheck disable=SC2206 # intentional word splitting of the wrapper prefix
wrap=(${SIDE_EFFECT_WRAP:-})

git cat-file -e "$sha^{commit}" 2>/dev/null || { echo "error: $sha is not a commit in this checkout" >&2; exit 1; }
sha="$(git rev-parse "$sha^{commit}")"
git fetch --tags --force --quiet origin 2>/dev/null || echo "note: could not fetch tags from origin"

if existing="$(git rev-parse -q --verify "refs/tags/$tag^{commit}")"; then
  if [ "$existing" = "$sha" ]; then
    echo "tag $tag is already on $sha: skipping tag creation"
  else
    echo "error: tag $tag points to $existing, not $sha" >&2
    exit 1
  fi
else
  git tag "$tag" "$sha"
  "${wrap[@]}" git push origin "refs/tags/$tag"
fi

if out="$(gh release view "$tag" 2>&1)"; then
  echo "release $tag already exists: skipping"
elif grep -qi "not found" <<<"$out"; then
  "${wrap[@]}" gh release create "$tag" --verify-tag --generate-notes --title "$tag"
else
  echo "error: cannot determine whether release $tag exists: $out" >&2
  exit 1
fi
