#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$ROOT_DIR/ARTIFACTS/validation}"
LANE_FILE="$ARTIFACT_DIR/infra_lane.json"
IMAGE_TAG="${V6_ALGO_TAG:-validation-${RUN_ID}}"
IMAGE_REPO="${V6_ALGO_REPO:-localhost:5001/strata-fit-v6-meta-algo}"
IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"
INFRA_PROFILE="${V6_INFRA_PROFILE:-baseline}"
INFRA_ENV_DIR="${V6_INFRA_ENV_DIR:-/tmp/strata-meta-infra-${RUN_ID}}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-${PYTHON_BIN:-python3}}"

mkdir -p "$ARTIFACT_DIR/scenarios"
start_epoch="$(date +%s)"
status="fail"
error_summary="lane failed"
scenario_results=()

cleanup() {
  end_epoch="$(date +%s)"
  duration="$((end_epoch - start_epoch))"
  {
    echo '{'
    echo '  "lane_name": "infra",'
    echo "  \"status\": \"$status\","
    echo "  \"duration_s\": $duration,"
    echo "  \"image\": \"$IMAGE\","
    echo "  \"profile\": \"$INFRA_PROFILE\","
    echo "  \"env_dir\": \"$INFRA_ENV_DIR\","
    echo "  \"error_summary\": \"$error_summary\","
    echo '  "scenarios": ['
    if [ "${#scenario_results[@]}" -gt 0 ]; then
      local first=true
      for item in "${scenario_results[@]}"; do
        if [ "$first" = true ]; then
          first=false
        else
          echo ','
        fi
        printf '    %s' "$item"
      done
      echo
    fi
    echo '  ]'
    echo '}'
  } > "$LANE_FILE"
}
trap cleanup EXIT
trap 'error_summary="infra smoke execution failed"' ERR

bootstrap_infra_env() {
  if [ -x "${PYTHON_BIN:-}" ]; then
    return
  fi

  PYTHON_BOOTSTRAP="$PYTHON_BOOTSTRAP" \
    bash "$ROOT_DIR/scripts/ci/bootstrap_meta_env.sh" "$INFRA_ENV_DIR"
  PYTHON_BIN="$INFRA_ENV_DIR/bin/python"
}

run_scenario() {
  local name="$1"
  local node_count="$2"
  local patients_per_node="$3"
  local run_linear="$4"
  local run_cox="$5"
  local run_km="$6"
  local started
  started="$(date +%s)"
  local artifact="$ARTIFACT_DIR/scenarios/${name}.json"
  local manifest="$ARTIFACT_DIR/scenarios/${name}_manifest.json"

  local skip_build="true"
  if [ "${#scenario_results[@]}" -eq 0 ]; then
    skip_build="false"
  fi

  if V6_SCENARIO_NAME="$name" \
    PYTHON_BIN="$PYTHON_BIN" \
    V6_NODE_COUNT="$node_count" \
    V6_PATIENTS_PER_NODE="$patients_per_node" \
    V6_RUN_LINEAR="$run_linear" \
    V6_RUN_COX="$run_cox" \
    V6_RUN_KM="$run_km" \
    V6_SKIP_BUILD_PUSH="$skip_build" \
    V6_ALGO_TAG="$IMAGE_TAG" \
    V6_RESULT_ARTIFACT="$artifact" \
    V6_DATA_MANIFEST_PATH="$manifest" \
    bash "$ROOT_DIR/tests/infra/run_local_infra_smoke.sh"; then
    local ended
    ended="$(date +%s)"
    scenario_results+=("{\"name\":\"$name\",\"status\":\"pass\",\"duration_s\":$((ended - started))}")
  else
    local ended
    ended="$(date +%s)"
    scenario_results+=("{\"name\":\"$name\",\"status\":\"fail\",\"duration_s\":$((ended - started))}")
    return 1
  fi
}

bootstrap_infra_env

run_scenario "baseline_3n_full" 3 36 true true true

if [ "$INFRA_PROFILE" = "full" ]; then
  run_scenario "fanout_5n_survival" 5 36 false true true
  run_scenario "fanout_8n_km" 8 24 false false true
  run_scenario "fanout_8n_cox" 8 24 false true false
fi

status="pass"
error_summary=""
