# strata-fit-v6-meta-algo-py

Standalone STRATA-FIT v6 meta-algorithm container for federated validation, imputation, and final-model orchestration.

## Runtime

- Base image: `python:3.11-slim`
- Container entrypoint: `python -m strata_fit_v6_meta_algo_py.container`
- Runtime contract: `RUN_CONTEXT_FILE`
- Public task method: `main`

This repo no longer depends on `vantage6-algorithm-tools` or Harbor `algorithm-base`. The package embeds its own `run_context` dispatcher, proxy client, JSON output helpers, and local in-process runner.

## Install

```bash
.venv/bin/python -m pip install -e .[dev]
```

The project keeps git and tarball dependencies commit-pinned. In CI and container builds we install conflict-prone dependencies with `--no-deps`, then add the explicit runtime transitive packages we actually need.

## Example Task Input

```python
input_ = {
    "method": "main",
    "kwargs": {
        "columns": ["DAS28", "CRP", "HAQ"],
        "run_validation": True,
        "imputation_strategy": "mean",
        "final_model": "sklearn_linear",
        "final_model_config": {
            "predictors": ["Age_diagnosis", "DAS28", "CRP"],
            "outcome": "RF_positivity",
            "n_local_iterations": 50,
            "model_kwargs": {"solver": "lbfgs"},
        },
    },
}
```

For raw STRATA-FIT survival flows:

```python
cox_config = {
    "time_col": "time",
    "outcome_col": "event",
    "expl_vars": ["Age_diagnosis", "DAS28", "CRP", "HAQ", "RF_positivity"],
    "max_iterations": 10,
    "tolerance": 1e-6,
    "preprocess_raw_data": True,
}

km_config = {
    "preprocess_raw_data": True,
}
```

## Local Development

Pure-Python local execution is available through `run_local_meta_algorithm(...)`. The in-process client preserves the same task fan-out contract used in federated execution, including the partial method names:

- `validate_partial`
- `imputation_compute_partial`
- `impute_and_train_sklearn_linear`
- `cox_get_unique_event_times_imputed`
- `cox_compute_summed_z_imputed`
- `cox_perform_iteration_imputed`
- `km_get_unique_event_times_imputed`
- `km_get_event_table_imputed`

Focused verification:

```bash
.venv/bin/python -m pytest \
  tests/test_runtime.py \
  tests/test_local_runtime.py \
  tests/test_meta_mock_client.py \
  tests/test_mock_pipeline.py -q
```

## Validation Harness

Validation artifacts are written under `ARTIFACTS/validation/`.

- `scripts/ci/clean_env_validate.sh`
  Fresh venv, deterministic install order, import checks, focused pytest lane.
- `scripts/ci/stress_matrix.py`
  Local scenario matrix without infrastructure.
- `scripts/ci/run_infra_lane.sh`
  `v6-infrastructure-sh` validation lane. By default it runs the amd64 signoff baseline:
  - `baseline_3n_full`
  Set `V6_INFRA_PROFILE=full` to expand back to:
  - `fanout_5n_survival`
  - `fanout_8n_km`
  - `fanout_8n_cox`
- `scripts/ci/run_full_validation.sh`
  Runs all lanes and writes `report.json` plus `report.md`.

Both `clean_env_validate.sh` and `run_infra_lane.sh` bootstrap their own disposable `/tmp` virtual environments so they do not mutate a working repo environment.

## Infrastructure Smoke

The infra lane uses the external harness repo at `../v6-infrastructure-sh` by default and treats amd64 CI as the authoritative signoff environment.

Quick start:

```bash
tests/infra/run_local_infra_smoke.sh
```

Useful overrides:

```bash
INFRA_DIR=/path/to/v6-infrastructure-sh \
PYTHON_BIN=/path/to/venv/bin/python \
V6_NODE_COUNT=5 \
V6_PATIENTS_PER_NODE=36 \
V6_LOCAL_REGISTRY_PORT=5002 \
V6_INFRA_PROFILE=full \
V6_MIRROR_INFRA_IMAGES=true \
DOCKER_REGISTRY=localhost:5002/v6infra \
V6_ALGO_TAG=dev \
tests/infra/run_local_infra_smoke.sh
```

If infra tasks fail, inspect the master org node container first; that is the fastest place to see the real Python traceback from the algorithm process.

On arm64 developer machines, local infra should be considered best-effort only. The published `server-lite` and `node-lite` images are signoff-tested on amd64 CI.
