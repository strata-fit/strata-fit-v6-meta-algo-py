#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$ROOT_DIR/ARTIFACTS/validation}"
VALIDATION_ENV_DIR="${V6_VALIDATION_ENV_DIR:-/tmp/strata-meta-validation-${RUN_ID}}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-${PYTHON_BIN:-python3}}"

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

ensure_validation_python() {
  if [ -n "${PYTHON_BIN:-}" ] && "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1; then
import pandas
import strata_fit_v6_meta_algo_py
PY
    return
  fi

  PYTHON_BOOTSTRAP="$PYTHON_BOOTSTRAP" \
    bash "$ROOT_DIR/scripts/ci/bootstrap_meta_env.sh" "$VALIDATION_ENV_DIR"
  PYTHON_BIN="$VALIDATION_ENV_DIR/bin/python"
}

ensure_validation_python

run_lane "clean_env" env RUN_ID="$RUN_ID" ARTIFACT_DIR="$ARTIFACT_DIR" PYTHON_BOOTSTRAP="$PYTHON_BIN" bash "$ROOT_DIR/scripts/ci/clean_env_validate.sh"
run_lane "stress_matrix" env RUN_ID="$RUN_ID" ARTIFACT_DIR="$ARTIFACT_DIR" PYTHON_BIN="$PYTHON_BIN" "$PYTHON_BIN" "$ROOT_DIR/scripts/ci/stress_matrix.py"
run_lane "security" env RUN_ID="$RUN_ID" ARTIFACT_DIR="$ARTIFACT_DIR" SECURITY_ARTIFACT_DIR="$ROOT_DIR/ARTIFACTS/security" PYTHON_BOOTSTRAP="$PYTHON_BIN" bash "$ROOT_DIR/scripts/ci/security_scan.sh"
run_lane "infra" env RUN_ID="$RUN_ID" ARTIFACT_DIR="$ARTIFACT_DIR" PYTHON_BIN="$PYTHON_BIN" bash "$ROOT_DIR/scripts/ci/run_infra_lane.sh"

"$PYTHON_BIN" "$ROOT_DIR/scripts/ci/report_summary.py"
