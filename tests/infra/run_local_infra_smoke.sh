#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Default assumes a sibling checkout like:
#   /path/to/strata-fit-v6-meta-algo-py
#   /path/to/v6-infrastructure-sh
# Override INFRA_DIR explicitly when your workspace layout differs.
INFRA_DIR="${INFRA_DIR:-$ROOT_DIR/../v6-infrastructure-sh}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TESTED_INFRA_COMMIT="${V6_TESTED_INFRA_COMMIT:-3133deb74a30fe34617d69d94628bbff38c71869}"
TESTED_VERSION_VANTAGE6="${V6_TESTED_VERSION_VANTAGE6:-4.14.0}"
TESTED_INFRA_IMAGE_TAG="${V6_TESTED_INFRA_IMAGE_TAG:-4.14.0-rc8}"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

NODE_COUNT="${V6_NODE_COUNT:-3}"
SCENARIO_NAME="${V6_SCENARIO_NAME:-dashboard_full_baseline}"
REGISTRY_PORT="${V6_LOCAL_REGISTRY_PORT:-5001}"
COLLAB_NAME="${V6_COLLABORATION_NAME:-meta-ci}"
TASK_TIMEOUT_S="${V6_TASK_TIMEOUT_S:-1200}"
SKIP_BUILD_PUSH="${V6_SKIP_BUILD_PUSH:-false}"
IMAGE_REPO="localhost:${REGISTRY_PORT}/strata-fit-v6-meta-algo"
IMAGE_TAG="${V6_ALGO_TAG:-local}"
IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"
REGISTRY_CONTAINER_NAME="v6-local-registry-meta-${REGISTRY_PORT}"
REGISTRY_STARTED=false
HOST_ARCH="$(uname -m)"
DOCKER_REGISTRY_WAS_SET=false
if [ "${DOCKER_REGISTRY+x}" = x ]; then
  DOCKER_REGISTRY_WAS_SET=true
fi

if [[ "$PYTHON_BIN" != */* ]]; then
  RESOLVED_PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
  if [ -n "$RESOLVED_PYTHON_BIN" ]; then
    PYTHON_BIN="$RESOLVED_PYTHON_BIN"
  fi
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python interpreter not found/executable: $PYTHON_BIN" >&2
  exit 1
fi

python_is_usable() {
  local candidate="$1"
  [ -n "$candidate" ] || return 1
  [ -x "$candidate" ] || return 1
  "$candidate" - <<PY >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (${MIN_PYTHON_MAJOR}, ${MIN_PYTHON_MINOR}) else 1)
PY
}

pick_fallback_python() {
  local candidate=""

  for candidate in \
    "$(command -v python3 2>/dev/null || true)" \
    "/home/debian/.pyenv/versions/3.12.0/bin/python" \
    "/home/debian/.pyenv/versions/3.11.0/bin/python" \
    "/home/debian/.pyenv/versions/3.10.0/bin/python" \
    "/usr/bin/python3"; do
    if python_is_usable "$candidate"; then
      printf '%s' "$candidate"
      return 0
    fi
  done

  return 1
}

if ! python_is_usable "$PYTHON_BIN"; then
  FALLBACK_PYTHON_BIN="$(pick_fallback_python || true)"
  if [ -n "$FALLBACK_PYTHON_BIN" ]; then
    echo "[meta-smoke][warn] Selected interpreter '$PYTHON_BIN' is not runnable or is below Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}; falling back to '$FALLBACK_PYTHON_BIN'" >&2
    PYTHON_BIN="$FALLBACK_PYTHON_BIN"
  else
    echo "No runnable Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} found for smoke bootstrap" >&2
    exit 1
  fi
fi

if [[ "$PYTHON_BIN" == */bin/* ]]; then
  PYTHON_ENV_ROOT="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
else
  PYTHON_ENV_ROOT="$INFRA_DIR/infrastructure/venv"
fi

if [ ! -d "$INFRA_DIR/infrastructure" ]; then
  cat >&2 <<EOF
v6-infrastructure-sh not found at: $INFRA_DIR

Before running local infra smoke tests, make sure:
  1. the harness repo is cloned or updated locally
  2. INFRA_DIR points at that checkout

Example:
  git clone https://github.com/mdw-nl/v6-infrastructure-sh.git /path/to/v6-infrastructure-sh
  INFRA_DIR=/path/to/v6-infrastructure-sh tests/infra/run_local_infra_smoke.sh

Notes:
  - this script reads INFRA_DIR
  - current ROOT_DIR is: $ROOT_DIR
EOF
  exit 1
fi

warn_if_version_drift() {
  local current_infra_commit=""

  if git -C "$INFRA_DIR" rev-parse HEAD >/dev/null 2>&1; then
    current_infra_commit="$(git -C "$INFRA_DIR" rev-parse HEAD)"
    if [ "$current_infra_commit" != "$TESTED_INFRA_COMMIT" ]; then
      cat >&2 <<EOF
[meta-smoke] advisory: local v6-infrastructure-sh checkout differs from the commit used in the tested CI lane.
  current: $current_infra_commit
  tested : $TESTED_INFRA_COMMIT

The run will continue, but if infra behaves unexpectedly, first retry with:
  git -C "$INFRA_DIR" fetch origin
  git -C "$INFRA_DIR" checkout "$TESTED_INFRA_COMMIT"
EOF
    fi
  fi

  if [ "${VERSION_VANTAGE6:-}" != "$TESTED_VERSION_VANTAGE6" ]; then
    echo "[meta-smoke] advisory: VERSION_VANTAGE6=${VERSION_VANTAGE6:-unset} but the tested baseline used ${TESTED_VERSION_VANTAGE6}" >&2
  fi

  if [ "${V6_SERVER_IMAGE_TAG:-}" != "$TESTED_INFRA_IMAGE_TAG" ]; then
    echo "[meta-smoke] advisory: V6_SERVER_IMAGE_TAG=${V6_SERVER_IMAGE_TAG:-unset} but the tested baseline used ${TESTED_INFRA_IMAGE_TAG}" >&2
  fi

  if [ "${V6_NODE_IMAGE_TAG:-}" != "$TESTED_INFRA_IMAGE_TAG" ]; then
    echo "[meta-smoke] advisory: V6_NODE_IMAGE_TAG=${V6_NODE_IMAGE_TAG:-unset} but the tested baseline used ${TESTED_INFRA_IMAGE_TAG}" >&2
  fi

  if [ "${V6_UI_IMAGE_TAG:-}" != "$TESTED_INFRA_IMAGE_TAG" ]; then
    echo "[meta-smoke] advisory: V6_UI_IMAGE_TAG=${V6_UI_IMAGE_TAG:-unset} but the tested baseline used ${TESTED_INFRA_IMAGE_TAG}" >&2
  fi
}

TMP_DIR="$(mktemp -d)"
DATA_DIR="$TMP_DIR/data"
NODES_ENV="$TMP_DIR/nodes.env"
DATA_MANIFEST_PATH="${V6_DATA_MANIFEST_PATH:-$TMP_DIR/run_manifest.json}"
RESULT_ARTIFACT="${V6_RESULT_ARTIFACT:-}"
BOOTSTRAP_ENV_DIR="$TMP_DIR/meta-smoke-env"
PYTHON_INTERPRETER="$PYTHON_BIN"
VENV_PATH="$PYTHON_ENV_ROOT"
NODES_CONFIG="$NODES_ENV"
COLLABORATION_NAME="$COLLAB_NAME"
ENVIRONMENT="${ENVIRONMENT:-CI}"
UI_ENABLED="${UI_ENABLED:-false}"
SERVER_URL="${SERVER_URL:-http://host.docker.internal}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
STRICT_DATA_CHECKS="${STRICT_DATA_CHECKS:-true}"
V6_SERVER_IMAGE_NAME="${V6_SERVER_IMAGE_NAME:-server-lite}"
V6_SERVER_IMAGE_TAG="${V6_SERVER_IMAGE_TAG:-4.14.0-rc8}"
V6_NODE_IMAGE_NAME="${V6_NODE_IMAGE_NAME:-node-lite}"
V6_NODE_IMAGE_TAG="${V6_NODE_IMAGE_TAG:-4.14.0-rc8}"
V6_UI_IMAGE_NAME="${V6_UI_IMAGE_NAME:-ui}"
V6_UI_IMAGE_TAG="${V6_UI_IMAGE_TAG:-4.14.0-rc8}"
trap 'set +e; cd "$INFRA_DIR/infrastructure" && ENVIRONMENT="$ENVIRONMENT" UI_ENABLED="$UI_ENABLED" NODES_CONFIG="$NODES_CONFIG" COLLABORATION_NAME="$COLLABORATION_NAME" PYTHON_INTERPRETER="$PYTHON_INTERPRETER" VENV_PATH="$VENV_PATH" SERVER_URL="$SERVER_URL" DOCKER_REGISTRY="$DOCKER_REGISTRY" STRICT_DATA_CHECKS="$STRICT_DATA_CHECKS" ./infra.sh down >/dev/null 2>&1 || true; if [ "$REGISTRY_STARTED" = true ]; then docker rm -f "$REGISTRY_CONTAINER_NAME" >/dev/null 2>&1 || true; fi; rm -rf "$TMP_DIR"' EXIT

SERVER_SOURCE_IMAGE="${V6_SERVER_SOURCE_IMAGE:-ghcr.io/mdw-nl/vantage6/infrastructure/server-lite:4.14.0-rc8}"
NODE_SOURCE_IMAGE="${V6_NODE_SOURCE_IMAGE:-ghcr.io/mdw-nl/vantage6/infrastructure/node-lite:4.14.0-rc8}"
UI_SOURCE_IMAGE="${V6_UI_SOURCE_IMAGE:-ghcr.io/mdw-nl/vantage6/infrastructure/ui:4.14.0-rc8}"
INFRA_SOURCE_PLATFORM="${V6_INFRA_SOURCE_PLATFORM:-linux/amd64}"
MIRROR_INFRA_IMAGES="${V6_MIRROR_INFRA_IMAGES:-false}"

python_has_smoke_dependencies() {
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import numpy
import pandas
import requests
import jwt
import pydantic
import sklearn
PY
}

ensure_python_ready() {
  if python_has_smoke_dependencies; then
    return
  fi

  echo "[meta-smoke] selected interpreter lacks smoke-test dependencies; bootstrapping disposable env in $BOOTSTRAP_ENV_DIR"
  PYTHON_BOOTSTRAP="$PYTHON_BIN" \
    bash "$ROOT_DIR/scripts/ci/bootstrap_meta_env.sh" "$BOOTSTRAP_ENV_DIR"
  PYTHON_BIN="$BOOTSTRAP_ENV_DIR/bin/python"
  PYTHON_ENV_ROOT="$BOOTSTRAP_ENV_DIR"
  PYTHON_INTERPRETER="$PYTHON_BIN"
  VENV_PATH="$PYTHON_ENV_ROOT"
}

host_requires_amd64_override() {
  case "$HOST_ARCH" in
    x86_64|amd64)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

is_localhost_registry() {
  local registry_ref="$1"

  case "$registry_ref" in
    localhost:*|127.0.0.1:*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

docker_image_exists() {
  local image_ref="$1"
  docker image inspect "$image_ref" >/dev/null 2>&1
}

pull_infra_source_image() {
  local image_ref="$1"

  if host_requires_amd64_override; then
    echo "[meta-smoke] pulling infra source image with platform ${INFRA_SOURCE_PLATFORM}: $image_ref"
    docker pull --platform "$INFRA_SOURCE_PLATFORM" "$image_ref"
  else
    echo "[meta-smoke] pulling infra source image: $image_ref"
    docker pull "$image_ref"
  fi
}

prepare_infra_image_ref() {
  local source_image="$1"
  local target_image="$2"
  local image_label="$3"

  if docker_image_exists "$target_image"; then
    echo "[meta-smoke] reusing prepared ${image_label} image: $target_image"
    return
  fi

  if [ "$source_image" != "$target_image" ] && docker_image_exists "$source_image"; then
    echo "[meta-smoke] tagging local ${image_label} image '$source_image' as '$target_image'"
  else
    pull_infra_source_image "$source_image"
  fi

  if [ "$source_image" != "$target_image" ]; then
    docker tag "$source_image" "$target_image"
  fi
}

ensure_python_ready

echo "[meta-smoke] generating synthetic data"
"$PYTHON_BIN" "$ROOT_DIR/tests/infra/prepare_meta_smoke_data.py" \
  --output-dir "$DATA_DIR" \
  --node-count "$NODE_COUNT" \
  --patients-per-node "${V6_PATIENTS_PER_NODE:-36}" \
  --manifest-path "$DATA_MANIFEST_PATH" \
  --scenario "$SCENARIO_NAME"

echo "[meta-smoke] generating nodes.env"
"$ROOT_DIR/tests/infra/generate_nodes_env.sh" "$NODE_COUNT" "$DATA_DIR" "$NODES_ENV"

set -a
# shellcheck source=/dev/null
source "$ROOT_DIR/tests/infra/config.env"
set +a

warn_if_version_drift

PYTHON_INTERPRETER="$PYTHON_BIN"
VENV_PATH="$PYTHON_ENV_ROOT"
NODES_CONFIG="$NODES_ENV"
COLLABORATION_NAME="$COLLAB_NAME"
ENVIRONMENT="${ENVIRONMENT:-CI}"
UI_ENABLED="${UI_ENABLED:-false}"
SERVER_URL="${SERVER_URL:-http://host.docker.internal}"
if [ "$DOCKER_REGISTRY_WAS_SET" = false ]; then
  DOCKER_REGISTRY="localhost:${REGISTRY_PORT}/v6infra"
fi
STRICT_DATA_CHECKS="${STRICT_DATA_CHECKS:-true}"

echo "[meta-smoke] ensuring local registry on port ${REGISTRY_PORT}"
if docker ps --format '{{.Ports}}' | grep -q ":${REGISTRY_PORT}->5000/tcp"; then
  echo "[meta-smoke] reusing existing registry bound to port ${REGISTRY_PORT}"
else
  docker run -d --restart unless-stopped -p "${REGISTRY_PORT}:5000" --name "$REGISTRY_CONTAINER_NAME" registry:2 >/dev/null
  REGISTRY_STARTED=true
fi

if [ "$MIRROR_INFRA_IMAGES" = "true" ] || is_localhost_registry "$DOCKER_REGISTRY"; then
  echo "[meta-smoke] preparing Vantage6 infra image refs under '$DOCKER_REGISTRY'"
  prepare_infra_image_ref "$SERVER_SOURCE_IMAGE" "${DOCKER_REGISTRY}/${V6_SERVER_IMAGE_NAME}:${V6_SERVER_IMAGE_TAG}" "server"
  prepare_infra_image_ref "$NODE_SOURCE_IMAGE" "${DOCKER_REGISTRY}/${V6_NODE_IMAGE_NAME}:${V6_NODE_IMAGE_TAG}" "node"

  if [ "$UI_ENABLED" = "true" ]; then
    prepare_infra_image_ref "$UI_SOURCE_IMAGE" "${DOCKER_REGISTRY}/${V6_UI_IMAGE_NAME}:${V6_UI_IMAGE_TAG}" "ui"
  fi
fi

cd "$INFRA_DIR/infrastructure"
ENVIRONMENT="$ENVIRONMENT" \
UI_ENABLED="$UI_ENABLED" \
NODES_CONFIG="$NODES_CONFIG" \
COLLABORATION_NAME="$COLLABORATION_NAME" \
PYTHON_INTERPRETER="$PYTHON_INTERPRETER" \
VENV_PATH="$VENV_PATH" \
SERVER_URL="$SERVER_URL" \
DOCKER_REGISTRY="$DOCKER_REGISTRY" \
V6_SERVER_IMAGE_NAME="$V6_SERVER_IMAGE_NAME" \
V6_SERVER_IMAGE_TAG="$V6_SERVER_IMAGE_TAG" \
V6_NODE_IMAGE_NAME="$V6_NODE_IMAGE_NAME" \
V6_NODE_IMAGE_TAG="$V6_NODE_IMAGE_TAG" \
V6_UI_IMAGE_NAME="$V6_UI_IMAGE_NAME" \
V6_UI_IMAGE_TAG="$V6_UI_IMAGE_TAG" \
STRICT_DATA_CHECKS="$STRICT_DATA_CHECKS" \
./infra.sh preflight
ENVIRONMENT="$ENVIRONMENT" \
UI_ENABLED="$UI_ENABLED" \
NODES_CONFIG="$NODES_CONFIG" \
COLLABORATION_NAME="$COLLABORATION_NAME" \
PYTHON_INTERPRETER="$PYTHON_INTERPRETER" \
VENV_PATH="$VENV_PATH" \
SERVER_URL="$SERVER_URL" \
DOCKER_REGISTRY="$DOCKER_REGISTRY" \
V6_SERVER_IMAGE_NAME="$V6_SERVER_IMAGE_NAME" \
V6_SERVER_IMAGE_TAG="$V6_SERVER_IMAGE_TAG" \
V6_NODE_IMAGE_NAME="$V6_NODE_IMAGE_NAME" \
V6_NODE_IMAGE_TAG="$V6_NODE_IMAGE_TAG" \
V6_UI_IMAGE_NAME="$V6_UI_IMAGE_NAME" \
V6_UI_IMAGE_TAG="$V6_UI_IMAGE_TAG" \
STRICT_DATA_CHECKS="$STRICT_DATA_CHECKS" \
./infra.sh up

if [ "$SKIP_BUILD_PUSH" = "true" ]; then
  echo "[meta-smoke] skipping build/push (V6_SKIP_BUILD_PUSH=true), using existing image: $IMAGE"
else
  echo "[meta-smoke] building and pushing algorithm image: $IMAGE"
  cd "$ROOT_DIR"
  docker build -t "$IMAGE" .
  docker push "$IMAGE"
fi

echo "[meta-smoke] running algorithm smoke tasks"
V6_ALGO_IMAGE="$IMAGE" \
V6_NODE_COUNT="$NODE_COUNT" \
V6_COLLABORATION_NAME="$COLLAB_NAME" \
V6_TASK_TIMEOUT_S="$TASK_TIMEOUT_S" \
V6_DATA_DIR="$DATA_DIR" \
V6_DATA_MANIFEST_PATH="$DATA_MANIFEST_PATH" \
V6_RESULT_ARTIFACT="$RESULT_ARTIFACT" \
V6_SCENARIO_NAME="$SCENARIO_NAME" \
"$PYTHON_BIN" "$ROOT_DIR/tests/infra/run_algo_smoke.py"

cd "$INFRA_DIR/infrastructure"
ENVIRONMENT="$ENVIRONMENT" \
UI_ENABLED="$UI_ENABLED" \
NODES_CONFIG="$NODES_CONFIG" \
COLLABORATION_NAME="$COLLABORATION_NAME" \
PYTHON_INTERPRETER="$PYTHON_INTERPRETER" \
VENV_PATH="$VENV_PATH" \
SERVER_URL="$SERVER_URL" \
DOCKER_REGISTRY="$DOCKER_REGISTRY" \
V6_SERVER_IMAGE_NAME="$V6_SERVER_IMAGE_NAME" \
V6_SERVER_IMAGE_TAG="$V6_SERVER_IMAGE_TAG" \
V6_NODE_IMAGE_NAME="$V6_NODE_IMAGE_NAME" \
V6_NODE_IMAGE_TAG="$V6_NODE_IMAGE_TAG" \
V6_UI_IMAGE_NAME="$V6_UI_IMAGE_NAME" \
V6_UI_IMAGE_TAG="$V6_UI_IMAGE_TAG" \
STRICT_DATA_CHECKS="$STRICT_DATA_CHECKS" \
./infra.sh test
ENVIRONMENT="$ENVIRONMENT" \
UI_ENABLED="$UI_ENABLED" \
NODES_CONFIG="$NODES_CONFIG" \
COLLABORATION_NAME="$COLLABORATION_NAME" \
PYTHON_INTERPRETER="$PYTHON_INTERPRETER" \
VENV_PATH="$VENV_PATH" \
SERVER_URL="$SERVER_URL" \
DOCKER_REGISTRY="$DOCKER_REGISTRY" \
V6_SERVER_IMAGE_NAME="$V6_SERVER_IMAGE_NAME" \
V6_SERVER_IMAGE_TAG="$V6_SERVER_IMAGE_TAG" \
V6_NODE_IMAGE_NAME="$V6_NODE_IMAGE_NAME" \
V6_NODE_IMAGE_TAG="$V6_NODE_IMAGE_TAG" \
V6_UI_IMAGE_NAME="$V6_UI_IMAGE_NAME" \
V6_UI_IMAGE_TAG="$V6_UI_IMAGE_TAG" \
STRICT_DATA_CHECKS="$STRICT_DATA_CHECKS" \
./infra.sh down

if [ "$REGISTRY_STARTED" = true ]; then
  docker rm -f "$REGISTRY_CONTAINER_NAME" >/dev/null || true
fi
echo "[meta-smoke] completed"
