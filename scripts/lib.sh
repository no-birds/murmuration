#!/usr/bin/env bash
# Shared helpers for the release scripts. Source it; do not execute it.

GROUP_PATH="org/nobirds/murmuration"
ARTIFACTS=(murmuration murmuration-gemini murmuration-openai murmuration-anthropic)
# Suffixes of the files every artifact version must serve (each also has a .asc signature).
FILE_SUFFIXES=(.pom .module .jar -sources.jar -javadoc.jar)

# Release semver only: no pre-release, no build metadata, no -SNAPSHOT, no leading zeros.
require_release_semver() {
  local v="${1:-}"
  if [[ ! "$v" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
    echo "error: '$v' is not a release semver (X.Y.Z)" >&2
    return 1
  fi
}

# Prints the repository-relative path of every file (and its signature) of a version, one per line.
artifact_files() {
  local version="$1" a s
  for a in "${ARTIFACTS[@]}"; do
    for s in "${FILE_SUFFIXES[@]}"; do
      echo "$GROUP_PATH/$a/$version/$a-$version$s"
      echo "$GROUP_PATH/$a/$version/$a-$version$s.asc"
    done
  done
}
