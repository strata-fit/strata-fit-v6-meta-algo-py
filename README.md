# strata-fit-v6-meta-algo-py

Meta-algorithm for vantage6 that orchestrates:

1. Optional data validation
2. Federated imputation metric fitting
3. Final federated model:
   - `sklearn_linear`
   - `cox`

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

For `final_model="cox"`, use:

```python
"final_model_config": {
    "time_col": "time",
    "outcome_col": "event",
    "expl_vars": ["x1", "x2"],
    "max_iterations": 10,
    "tolerance": 1e-6,
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

1. Generates synthetic node CSVs with columns needed for both final models.
2. Boots local vantage6 infra via the external `v6-infrastructure-sh` repo.
3. Builds and pushes a local registry image for this algorithm.
4. Submits and validates two tasks:
   - `final_model=sklearn_linear`
   - `final_model=cox`
5. Runs infra container smoke checks and tears everything down.

See `tests/infra/README.md` for overrides (`INFRA_DIR`, node count, image tag,
registry port, python interpreter, skip-build mode).
