#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$ROOT_DIR/ARTIFACTS/validation}"

mkdir -p "$ARTIFACT_DIR/scenarios"
echo "$RUN_ID" > "$ARTIFACT_DIR/run_id.txt"

run_lane() {
  local lane="$1"
  shift
  echo "[validation] running lane: $lane"
  if "$@"; then
    echo "[validation] lane passed: $lane"
  else
    echo "[validation] lane failed: $lane"
  fi
}

run_lane "clean_env" env RUN_ID="$RUN_ID" ARTIFACT_DIR="$ARTIFACT_DIR" bash "$ROOT_DIR/scripts/ci/clean_env_validate.sh"
run_lane "stress_matrix" env RUN_ID="$RUN_ID" ARTIFACT_DIR="$ARTIFACT_DIR" python3 "$ROOT_DIR/scripts/ci/stress_matrix.py"
run_lane "infra" env RUN_ID="$RUN_ID" ARTIFACT_DIR="$ARTIFACT_DIR" bash "$ROOT_DIR/scripts/ci/run_infra_lane.sh"

python3 "$ROOT_DIR/scripts/ci/report_summary.py"
