#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$ROOT_DIR/ARTIFACTS/validation}"
LANE_FILE="$ARTIFACT_DIR/clean_env_lane.json"
ENV_DIR="/tmp/strata-meta-clean-${RUN_ID}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-${PYTHON_BIN:-python3}}"

mkdir -p "$ARTIFACT_DIR"
start_epoch="$(date +%s)"
status="fail"
error_summary="lane failed"

cleanup() {
  end_epoch="$(date +%s)"
  duration="$((end_epoch - start_epoch))"
  cat > "$LANE_FILE" <<EOF
{
  "lane_name": "clean_env",
  "status": "$status",
  "duration_s": $duration,
  "env_dir": "$ENV_DIR",
  "error_summary": "$error_summary"
}
EOF
}
trap cleanup EXIT
trap 'error_summary="clean env validation failed"' ERR

rm -rf "$ENV_DIR"
PYTHON_BOOTSTRAP="$PYTHON_BOOTSTRAP" \
  bash "$ROOT_DIR/scripts/ci/bootstrap_meta_env.sh" "$ENV_DIR"

"$ENV_DIR/bin/python" - <<'PY'
import dynaconf
import strata_fit_v6_meta_algo_py
from strata_fit_v6_meta_algo_py import central as meta_central
from strata_fit_v6_meta_algo_py import methods as meta_methods
assert meta_central is not None
assert meta_methods is not None
print("imports_ok")
PY

"$ENV_DIR/bin/python" -m pip freeze > "$ARTIFACT_DIR/clean_env_pip_freeze.txt"
"$ENV_DIR/bin/python" -m pytest \
  "$ROOT_DIR/tests/test_runtime.py" \
  "$ROOT_DIR/tests/test_local_runtime.py" \
  "$ROOT_DIR/tests/test_meta_mock_client.py" \
  "$ROOT_DIR/tests/test_mock_pipeline.py" -q

status="pass"
error_summary=""
