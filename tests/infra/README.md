# Meta-algo infra smoke (`v6-infrastructure-sh`)

This folder contains reusable infra-smoke tooling for `strata-fit-v6-meta-algo-py`.
All infrastructure lifecycle operations are delegated to the external
`v6-infrastructure-sh` repository.

## Prerequisite: external infra repo

Clone the shared infrastructure harness:

```bash
cd ..
git clone https://github.com/mdw-nl/v6-infrastructure-sh.git
```

## Scripts

- `prepare_meta_smoke_data.py`:
  - Generates synthetic CSV buckets (`data_bucketN.csv`) with columns required by both:
    - `sklearn_linear`: `Age_diagnosis`, `DAS28`, `CRP`, `HAQ`, `RF_positivity`
    - `cox`: `time`, `event`, `x1`, `x2`
  - Injects missing values so imputation is exercised.
- `generate_nodes_env.sh`:
  - Creates `nodes.env` for 2-8 nodes using deterministic names/API keys.
- `run_algo_smoke.py`:
  - Submits two tasks (`sklearn_linear`, `cox`) and validates terminal state,
    decoded result payload, and child run completion.
- `run_local_infra_smoke.sh`:
  - End-to-end wrapper: preflight/up/build/push/task-smoke/infra-test/down.

## Quick start

From repo root:

```bash
tests/infra/run_local_infra_smoke.sh
```

Expected layout defaults:

- Algorithm repo: current directory
- Infra harness repo: `../v6-infrastructure-sh`
- Python: `../.venv/bin/python`

## Useful overrides

```bash
INFRA_DIR=/path/to/v6-infrastructure-sh \
PYTHON_BIN=/path/to/python \
V6_NODE_COUNT=4 \
V6_LOCAL_REGISTRY_PORT=5002 \
V6_SKIP_BUILD_PUSH=true \
V6_ALGO_TAG=dev \
V6_COLLABORATION_NAME=meta-ci \
tests/infra/run_local_infra_smoke.sh
```

Notes:

- The runner reuses an existing local registry bound to the selected port.
- The runner does not overwrite `v6-infrastructure-sh/infrastructure/config.env`.
- Set `V6_SKIP_BUILD_PUSH=true` to reuse an already-pushed local image tag.
