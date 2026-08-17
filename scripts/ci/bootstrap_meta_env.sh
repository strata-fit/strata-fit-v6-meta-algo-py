#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${1:-}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-${PYTHON_BIN:-python3}}"
V6_FEDERATED_CORE_PACKAGE_SPEC="${V6_FEDERATED_CORE_PACKAGE_SPEC:-v6-federated-algo-core-py @ https://github.com/mdw-nl/v6-federated-algo-core-v6/archive/c29dd63f40c6e3997a0865cb0cbc81dd9ce02a60.tar.gz}"
STRATA_FIT_DATA_VALIDATOR_PACKAGE_SPEC="${STRATA_FIT_DATA_VALIDATOR_PACKAGE_SPEC:-strata-fit-v6-data-validator-py @ https://github.com/strata-fit/strata-fit-data-schema/archive/c77d319b6539bdc48314738981b5bde478d2bacd.tar.gz}"

if [ -z "$ENV_DIR" ]; then
  echo "usage: $0 <env-dir>" >&2
  exit 1
fi

python_is_usable() {
  local candidate="$1"
  [ -n "$candidate" ] || return 1
  command -v "$candidate" >/dev/null 2>&1 || return 1
  "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

pick_python_bootstrap() {
  local candidate=""

  for candidate in \
    "$PYTHON_BOOTSTRAP" \
    "${PYTHON_BIN:-}" \
    python3.12 \
    python3.11 \
    python3.10 \
    python3 \
    python; do
    if python_is_usable "$candidate"; then
      printf '%s' "$candidate"
      return 0
    fi
  done

  return 1
}

PYTHON_BOOTSTRAP="$(pick_python_bootstrap || true)"
if [ -z "$PYTHON_BOOTSTRAP" ]; then
  echo "error: no runnable Python >= 3.10 found for bootstrap_meta_env.sh" >&2
  exit 1
fi

rm -rf "$ENV_DIR"
"$PYTHON_BOOTSTRAP" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

# Install explicitly managed dependencies first, then the editable package, so
# the final environment can be checked deterministically.
"$ENV_DIR/bin/python" -m pip install \
  pandas numpy==1.26.0 scipy scikit-learn \
  dynaconf pytest pytest-mock requests PyJWT pydantic \
  "vantage6-client==4.14.0"
"$ENV_DIR/bin/python" -m pip install --no-deps \
  "$V6_FEDERATED_CORE_PACKAGE_SPEC" \
  "$STRATA_FIT_DATA_VALIDATOR_PACKAGE_SPEC"
"$ENV_DIR/bin/python" -m pip install joblib threadpoolctl
"$ENV_DIR/bin/python" -m pip install --no-deps -e "$ROOT_DIR"
"$ENV_DIR/bin/python" - <<'PY'
import dynaconf
import strata_fit_v6_meta_algo_py
from strata_fit_v6_meta_algo_py import central as meta_central
from strata_fit_v6_meta_algo_py import methods as meta_methods

assert meta_central is not None
assert meta_methods is not None
print("imports_ok")
PY
