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
  - Generates synthetic STRATA-FIT `PatientData`-schema CSV buckets (`data_bucketN.csv`).
  - Data is reverse-engineered from a latent Cox-style population (`time/event/risk`), then expanded
    to longitudinal RA visits with D2T signal and controlled missingness.
- `generate_nodes_env.sh`:
  - Creates `nodes.env` for 2-8 nodes using deterministic names/API keys.
- `run_algo_smoke.py`:
  - Builds a mock baseline on the same CSV partitions.
  - Submits infra tasks (`cox`, `km`, optional `sklearn_linear`) and validates terminal state,
    decoded result payload, child run completion, and mock-vs-infra similarity for Cox/KM.
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
V6_PATIENTS_PER_NODE=40 \
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
- Set `V6_RUN_LINEAR=false` if you only want the raw-data survival pipeline (`cox` + `km`).
