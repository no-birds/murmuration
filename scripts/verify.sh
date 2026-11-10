#!/usr/bin/env bash
# Usage: verify.sh <version> present|absent
#
#   present - wait (bounded) until every file of the 4 artifacts is served anonymously.
#   absent  - one pass; fail if ANY file is served (the version number is burned) or if the
#             answer is neither 200 nor 404 (unknown state is not "absent").
#
# Environment:
#   MAVEN_BASE      base URL of the repository (default: Maven Central host used by Gradle consumers).
#                   The stage job points it at a local HTTP server over the staging directory.
#   VERIFY_TIMEOUT  seconds to wait in "present" mode (default 1800).
#   VERIFY_INTERVAL seconds between passes (default 30).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$here/lib.sh"

version="${1:?usage: verify.sh <version> present|absent}"
mode="${2:?usage: verify.sh <version> present|absent}"
base="${MAVEN_BASE:-https://repo.maven.apache.org/maven2}"
base="${base%/}"
timeout="${VERIFY_TIMEOUT:-1800}"
interval="${VERIFY_INTERVAL:-30}"

require_release_semver "$version"

status_of() {
  curl -s -o /dev/null -I -L --max-time 30 -w '%{http_code}' "$base/$1" || echo 000
}

case "$mode" in
  absent)
    bad=0
    while read -r path; do
      code="$(status_of "$path")"
      case "$code" in
        404) ;;
        200) echo "burned: $base/$path is served"; bad=1 ;;
        *) echo "unknown: $base/$path answered $code"; bad=1 ;;
      esac
    done < <(artifact_files "$version")
    [ "$bad" -eq 0 ] || exit 1
    echo "version $version is not served at $base"
    ;;
  present)
    deadline=$(( $(date +%s) + timeout ))
    while true; do
      missing=0
      while read -r path; do
        [ "$(status_of "$path")" = "200" ] || { missing=$((missing + 1)); last="$path"; }
      done < <(artifact_files "$version")
      if [ "$missing" -eq 0 ]; then
        echo "all files of $version are served at $base"
        exit 0
      fi
      if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "timeout: $missing file(s) of $version still missing at $base (last: $last)"
        exit 1
      fi
      echo "waiting: $missing file(s) missing (last: $last)"
      sleep "$interval"
    done
    ;;
  *)
    echo "mode must be present or absent" >&2
    exit 2
    ;;
esac
