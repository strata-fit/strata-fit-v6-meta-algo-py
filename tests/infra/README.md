# Meta-algo infra smoke (`v6-infrastructure-sh`)

This folder contains reusable infra-smoke tooling for `strata-fit-v6-meta-algo-py`.
All infrastructure lifecycle operations are delegated to the external
`v6-infrastructure-sh` repository.

## Prerequisite: external infra repo

Before using the quick start, make sure the external infrastructure harness
exists locally and that your paths match the script defaults.

The smoke wrapper assumes this layout unless you override `INFRA_DIR`:

```text
workspace/
  strata-fit-v6-meta-algo-py/
  v6-infrastructure-sh/
```

If your checkout lives elsewhere, set `INFRA_DIR` explicitly.

Clone or update the shared infrastructure harness:

```bash
cd ..
git clone https://github.com/mdw-nl/v6-infrastructure-sh.git
```

If you already have it:

```bash
cd /path/to/v6-infrastructure-sh
git pull
```

For the meta-algo baseline, the tested CI lane used:

- `v6-infrastructure-sh` commit `3133deb74a30fe34617d69d94628bbff38c71869`
- `VERSION_VANTAGE6=4.14.0`
- `server-lite`, `node-lite`, and `ui` image tag `4.14.0-rc8`

Local runs on different revisions are allowed, but treat them as drift from the
tested baseline and expect more debugging if behavior changes.

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
  - Uses GHCR infra images directly by default.
  - Supports optional local mirroring for best-effort arm64 developer runs.

The CI infra lane builds on this wrapper. The default signoff path is:

- `dashboard_full_baseline`

Set `V6_INFRA_PROFILE=full` to expand back to:

- `site_heterogeneity_5n`
- `fanout_8n_km`
- `fanout_8n_cox`

## Quick start

From repo root:

```bash
tests/infra/run_local_infra_smoke.sh
```

Before that command, verify these local prerequisites:

- `v6-infrastructure-sh` is cloned locally and `INFRA_DIR` points to it when it is not in `../v6-infrastructure-sh`
- Docker is available and running
- you have a usable Python interpreter for `PYTHON_BIN`; if it lacks the smoke dependencies, the wrapper bootstraps a disposable `/tmp` env automatically
- the harness dependencies in `v6-infrastructure-sh` are installed the way that repo expects
- if you want parity with the tested baseline, check out the tested harness commit and use the tested Vantage6/image versions

Expected layout defaults:

- Algorithm repo: current directory
- Infra harness repo: `../v6-infrastructure-sh`
- Python: provide `PYTHON_BIN` explicitly if you want a specific interpreter; otherwise the wrapper can bootstrap a disposable `/tmp` env when the selected interpreter is missing required deps

Typical local invocation when the harness is not a sibling checkout:

```bash
INFRA_DIR=/path/to/v6-infrastructure-sh \
PYTHON_BIN=/path/to/venv/bin/python \
tests/infra/run_local_infra_smoke.sh
```

If you want to align to the originally tested harness revision first:

```bash
git -C /path/to/v6-infrastructure-sh fetch origin
git -C /path/to/v6-infrastructure-sh checkout 3133deb74a30fe34617d69d94628bbff38c71869
```

## Useful overrides

```bash
INFRA_DIR=/path/to/v6-infrastructure-sh \
PYTHON_BIN=/path/to/venv/bin/python \
V6_NODE_COUNT=4 \
V6_PATIENTS_PER_NODE=40 \
V6_SCENARIO_NAME=site_heterogeneity_5n \
V6_LOCAL_REGISTRY_PORT=5001 \
V6_MIRROR_INFRA_IMAGES=true \
DOCKER_REGISTRY=localhost:5001/v6infra \
V6_ALGO_TAG=dev \
V6_COLLABORATION_NAME=meta-ci \
tests/infra/run_local_infra_smoke.sh
```

Notes:

- The runner reuses an existing local registry bound to the selected port.
- The default local registry port is `5001`; only change it if that port is already occupied.
- Set `V6_SCENARIO_NAME` to one of the stress scenarios to drive data generation and model config from the shared scenario manifest.
- The runner does not overwrite `v6-infrastructure-sh/infrastructure/config.env`.
- The runner warns, but does not fail, when your local harness commit or Vantage6/image versions drift from the tested baseline.
- If the selected `PYTHON_BIN` cannot import the smoke dependencies, the runner bootstraps a disposable env under `/tmp` before generating data.
- On hosts where the locally available `server-lite` / `node-lite` images are amd64-only and Docker cannot execute them, the harness now fails fast with an architecture probe error before server startup.
- Set `V6_SKIP_BUILD_PUSH=true` to reuse an already-pushed local image tag.
- Set `V6_RUN_LINEAR=false` if you only want the raw-data survival pipeline (`cox` + `km`).
- Authoritative infra validation is expected to run on amd64 CI.
- A local `exec format error` from the Vantage6 infra images is an environment/architecture problem, not a meta-algorithm problem.
- When a task fails in infra, start by attaching to the master org node container and checking the Python traceback there.
