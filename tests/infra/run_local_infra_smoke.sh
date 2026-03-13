#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INFRA_DIR="${INFRA_DIR:-$ROOT_DIR/../v6-infrastructure-sh}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/../.venv/bin/python}"

NODE_COUNT="${V6_NODE_COUNT:-3}"
REGISTRY_PORT="${V6_LOCAL_REGISTRY_PORT:-5001}"
COLLAB_NAME="${V6_COLLABORATION_NAME:-meta-ci}"
TASK_TIMEOUT_S="${V6_TASK_TIMEOUT_S:-1200}"
SKIP_BUILD_PUSH="${V6_SKIP_BUILD_PUSH:-false}"
IMAGE_REPO="localhost:${REGISTRY_PORT}/strata-fit-v6-meta-algo"
IMAGE_TAG="${V6_ALGO_TAG:-local}"
IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"
REGISTRY_CONTAINER_NAME="v6-local-registry-meta-${REGISTRY_PORT}"
REGISTRY_STARTED=false

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python interpreter not found/executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$PYTHON_BIN" == */bin/* ]]; then
  PYTHON_ENV_ROOT="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
else
  PYTHON_ENV_ROOT="$INFRA_DIR/infrastructure/venv"
fi

if [ ! -d "$INFRA_DIR/infrastructure" ]; then
  echo "v6-infrastructure-sh not found at: $INFRA_DIR" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
DATA_DIR="$TMP_DIR/data"
NODES_ENV="$TMP_DIR/nodes.env"
PYTHON_INTERPRETER="$PYTHON_BIN"
VENV_PATH="$PYTHON_ENV_ROOT"
NODES_CONFIG="$NODES_ENV"
COLLABORATION_NAME="$COLLAB_NAME"
ENVIRONMENT="${ENVIRONMENT:-CI}"
UI_ENABLED="${UI_ENABLED:-false}"
SERVER_URL="${SERVER_URL:-http://host.docker.internal}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-harbor2.vantage6.ai/infrastructure}"
STRICT_DATA_CHECKS="${STRICT_DATA_CHECKS:-true}"
trap 'set +e; cd "$INFRA_DIR/infrastructure" && ENVIRONMENT="$ENVIRONMENT" UI_ENABLED="$UI_ENABLED" NODES_CONFIG="$NODES_CONFIG" COLLABORATION_NAME="$COLLABORATION_NAME" PYTHON_INTERPRETER="$PYTHON_INTERPRETER" VENV_PATH="$VENV_PATH" SERVER_URL="$SERVER_URL" DOCKER_REGISTRY="$DOCKER_REGISTRY" STRICT_DATA_CHECKS="$STRICT_DATA_CHECKS" ./infra.sh down >/dev/null 2>&1 || true; if [ "$REGISTRY_STARTED" = true ]; then docker rm -f "$REGISTRY_CONTAINER_NAME" >/dev/null 2>&1 || true; fi; rm -rf "$TMP_DIR"' EXIT

echo "[meta-smoke] generating synthetic data"
"$PYTHON_BIN" "$ROOT_DIR/tests/infra/prepare_meta_smoke_data.py" \
  --output-dir "$DATA_DIR" \
  --node-count "$NODE_COUNT" \
  --rows 40

echo "[meta-smoke] generating nodes.env"
"$ROOT_DIR/tests/infra/generate_nodes_env.sh" "$NODE_COUNT" "$DATA_DIR" "$NODES_ENV"

set -a
# shellcheck source=/dev/null
source "$ROOT_DIR/tests/infra/config.env"
set +a

PYTHON_INTERPRETER="$PYTHON_BIN"
VENV_PATH="$PYTHON_ENV_ROOT"
NODES_CONFIG="$NODES_ENV"
COLLABORATION_NAME="$COLLAB_NAME"
ENVIRONMENT="${ENVIRONMENT:-CI}"
UI_ENABLED="${UI_ENABLED:-false}"
SERVER_URL="${SERVER_URL:-http://host.docker.internal}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-harbor2.vantage6.ai/infrastructure}"
STRICT_DATA_CHECKS="${STRICT_DATA_CHECKS:-true}"

cd "$INFRA_DIR/infrastructure"
ENVIRONMENT="$ENVIRONMENT" \
UI_ENABLED="$UI_ENABLED" \
NODES_CONFIG="$NODES_CONFIG" \
COLLABORATION_NAME="$COLLABORATION_NAME" \
PYTHON_INTERPRETER="$PYTHON_INTERPRETER" \
VENV_PATH="$VENV_PATH" \
SERVER_URL="$SERVER_URL" \
DOCKER_REGISTRY="$DOCKER_REGISTRY" \
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
STRICT_DATA_CHECKS="$STRICT_DATA_CHECKS" \
./infra.sh up

echo "[meta-smoke] ensuring local registry on port ${REGISTRY_PORT}"
if docker ps --format '{{.Ports}}' | grep -q ":${REGISTRY_PORT}->5000/tcp"; then
  echo "[meta-smoke] reusing existing registry bound to port ${REGISTRY_PORT}"
else
  docker run -d --restart unless-stopped -p "${REGISTRY_PORT}:5000" --name "$REGISTRY_CONTAINER_NAME" registry:2 >/dev/null
  REGISTRY_STARTED=true
fi

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
STRICT_DATA_CHECKS="$STRICT_DATA_CHECKS" \
./infra.sh down

if [ "$REGISTRY_STARTED" = true ]; then
  docker rm -f "$REGISTRY_CONTAINER_NAME" >/dev/null || true
fi
echo "[meta-smoke] completed"
