#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${1:-}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-${PYTHON_BIN:-python3}}"

if [ -z "$ENV_DIR" ]; then
  echo "usage: $0 <env-dir>" >&2
  exit 1
fi

rm -rf "$ENV_DIR"
"$PYTHON_BOOTSTRAP" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

# Install the target package first, then add explicitly managed dependencies
# to avoid resolver drift from mutable/transitive git metadata.
"$ENV_DIR/bin/python" -m pip install --no-deps -e "$ROOT_DIR"
"$ENV_DIR/bin/python" -m pip install \
  pandas numpy scipy scikit-learn \
  dynaconf pytest pytest-mock requests PyJWT pydantic \
  "vantage6-client==4.14.0"
"$ENV_DIR/bin/python" -m pip install --no-deps \
  "v6-federated-algo-core-py @ https://github.com/mdw-nl/v6-federated-algo-core-v6/archive/c29dd63f40c6e3997a0865cb0cbc81dd9ce02a60.tar.gz" \
  "strata-fit-v6-data-validator-py @ https://github.com/strata-fit/strata-fit-data-schema/archive/c77d319b6539bdc48314738981b5bde478d2bacd.tar.gz"
"$ENV_DIR/bin/python" -m pip install joblib threadpoolctl
