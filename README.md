# strata-fit-v6-meta-algo-py

Meta-algorithm for vantage6 that orchestrates:

1. Optional data validation
2. Federated imputation metric fitting
3. STRATA-FIT survival preprocessing (for Cox/KM when enabled)
4. Final federated model:
   - `sklearn_linear`
   - `cox`
   - `km`

The pipeline intentionally omits a stats-extraction stage for now.

## Supported entrypoint

- `main`

## Example task input

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

For `final_model="cox"` from raw STRATA-FIT tables, use:

```python
"final_model_config": {
    "time_col": "time",
    "outcome_col": "event",
    "expl_vars": ["Age_diagnosis", "DAS28", "CRP", "HAQ", "RF_positivity"],
    "max_iterations": 10,
    "tolerance": 1e-6,
    "preprocess_raw_data": True,
}
```

For `final_model="km"` from raw STRATA-FIT tables, use:

```python
"final_model_config": {
    "preprocess_raw_data": True,
}
```

## Manual Infra Smoke

Infrastructure smoke testing in this repo relies on the external harness:

- `https://github.com/mdw-nl/v6-infrastructure-sh`

Clone it as a sibling directory (default expected by our smoke script):

```bash
cd ..
git clone https://github.com/mdw-nl/v6-infrastructure-sh.git
```

From this repo root, run:

```bash
tests/infra/run_local_infra_smoke.sh
```

What this does:

1. Generates synthetic STRATA-FIT schema-compatible node CSVs (derived from latent Cox data).
2. Boots local vantage6 infra via the external `v6-infrastructure-sh` repo.
3. Builds and pushes a local registry image for this algorithm.
4. Runs local mock baselines on the exact same CSV partitions.
5. Submits and validates infra tasks:
   - `final_model=cox` (with preprocessing)
   - `final_model=km` (with preprocessing)
   - optional `final_model=sklearn_linear`
6. Compares Cox/KM infra outputs against mock outputs for parity.
7. Runs infra container smoke checks and tears everything down.

See `tests/infra/README.md` for overrides (`INFRA_DIR`, node count, image tag,
registry port, python interpreter, skip-build mode).
