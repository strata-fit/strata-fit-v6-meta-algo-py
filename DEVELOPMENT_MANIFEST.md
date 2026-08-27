# v6-meta-algo-py Development and Testing Manifest

## Scope
This manifest captures the repeated setup/debug issues encountered while validating the standalone meta-algorithm migration, and the deterministic workflow to avoid repeating them.

## Current State
- `strata-fit-v6-meta-algo-py` is already on standalone `run_context`.
- Clean-env validation, local stress-matrix validation, and the lightweight security lane are the default pre-infra gates.
- The authoritative signoff environment for infrastructure evaluation is amd64.
- KM preprocessing and Cox math used for signoff are internal modules in this repo; published external `km`/`cox` repos are no longer required for the signoff path.

## Root Causes of Repeated Setup Failures
1. Missing runtime deps in `pyproject.toml` for modules imported at runtime (`validator`, `core`, earlier `coxph`/`km` experiments).
2. Dependency resolver conflicts caused by mixed direct git refs and upstream package metadata.
3. Import-time side effects in validator package:
  - `strata_fit_v6_data_validator_py` imports `dynaconf` through `config.config` at import time.
4. Environment drift:
  - repeated installs introduced mixed `vantage6-algorithm-tools` versions (`4.13.3` vs `4.14.0`).
5. Native crash class:
  - segfault observed in `polars` during imputation (`group_by(...).agg(...).collect(...)`) in a mixed environment.

## Immutable Pins Used
These are the signoff-critical external pins still in use:

- `v6-federated-algo-core-v6`: `c29dd63f40c6e3997a0865cb0cbc81dd9ce02a60`
- `strata-fit-data-schema` (validator pkg): `c77d319b6539bdc48314738981b5bde478d2bacd`

## Important Limitation
The remaining external signoff path still depends on git-sourced packages rather than immutable PyPI releases.

Practical consequence:
- `pip install -e '.[dev]'` can still drift unless install order is controlled.
  - `pip install --no-deps` flows still need explicit stable runtime primitives before the pinned git/tar packages.

## Deterministic Install Workflow (Recommended)
Use explicit order and `--no-deps` for conflict-prone git packages.

1. Base editable install without resolver backtracking:
```bash
.venv/bin/python -m pip install --no-deps -e .
```

2. Install stable runtime/tooling primitives:
```bash
.venv/bin/python -m pip install \
  dynaconf requests PyJWT pydantic \
  pandas numpy scipy scikit-learn \
  pytest pytest-mock vantage6-client pip-audit bandit
```

3. Install pinned algorithm/runtime git deps without transitive re-resolution:
```bash
.venv/bin/python -m pip install --no-deps \
  "v6-federated-algo-core-py @ https://github.com/mdw-nl/v6-federated-algo-core-v6/archive/c29dd63f40c6e3997a0865cb0cbc81dd9ce02a60.tar.gz" \
  "strata-fit-v6-data-validator-py @ https://github.com/strata-fit/strata-fit-data-schema/archive/c77d319b6539bdc48314738981b5bde478d2bacd.tar.gz"
```

4. Verify critical imports before tests:
```bash
.venv/bin/python - <<'PY'
import dynaconf
import strata_fit_v6_data_validator_py
import strata_fit_v6_meta_algo_py
print("imports_ok")
PY
```

5. Run mock test subset:
```bash
.venv/bin/python -m pytest tests/test_d2t_preprocessing.py tests/test_meta_mock_client.py tests/test_mock_pipeline.py -q
```

## D2T Event Correctness
- D2T RA event labeling is a correctness gate for Cox/KM, not a plotting detail.
- The operational definition remains all three criteria together:
  - at least two unique b/tsDMARD classes and at least six months since last DMARD change
  - rolling DAS28 > 3.2 or rolling CRP > 1.0
  - Pat_global > 50 or Ph_global > 50
- Do not use derived fields (`D2T_RA`, `D2T_crit*`, `TTE`, `event_type`, `interval_start`, `interval_end`, `time`, `event`) as raw predictors in stress scenarios.
- Scenario artifacts include per-node counts for `D2T_crit1`, `D2T_crit2`, `D2T_crit3`, and all-three `D2T_RA`.

## Debug Checklist for Future Agents
- Confirm branch and dirty state first:
  - `git status --short --branch`
- If failures are `ModuleNotFoundError`:
  - do not patch code first; patch environment with deterministic install order.
- If failures are `ValidationError` on `imputation_compute_partial` output:
  - check `imputation_compute_partial_handler` return normalization (`dict` pass-through, dataframe-like `.to_dict()`).
- If downstream handlers receive dicts where DataFrames are expected:
  - check `_impute_locally` normalization to DataFrame.
- If infra validation fails on arm64:
  - do not treat that as a signoff blocker.
  - rerun on amd64 CI before changing the algorithm path.
- If task execution fails in infra:
  - inspect the master organization node container traceback before editing harness config.
- If stress validation fails:
  - inspect `ARTIFACTS/validation/meta_stress/<scenario>.json` before rerunning infra.
- If security validation fails:
  - inspect `ARTIFACTS/security/report.json`; fix static regressions and direct high-impact findings before dependency refreshes.

## Security/Reproducibility Guidance
- Prefer commit-pinned git/tar refs over branch refs whenever upstream allows.
- Use `--no-deps` on conflict-prone git packages to avoid resolver drift in deterministic builds.
- During debugging, a temporary full resolver install can still be informative to expose hidden transitive conflicts.
- Record exact commit SHAs and install order in this manifest when changing dependencies.
- Long-term fix required upstream:
  - publish immutable PyPI releases for `v6-federated-algo-core-py` and validator/imputation artifacts.
  - remove mutable branch-based direct URLs from package metadata across remaining shared repos.

## Why Pyproject And Docker Both Pin
- `pyproject.toml` pins make local editable installs reproducible for developers and CI that installs from source.
- Docker pins make image builds deterministic even when install order uses explicit `pip install --no-deps` layering.
- They must stay aligned; update both in the same change whenever dependency SHAs are changed.
