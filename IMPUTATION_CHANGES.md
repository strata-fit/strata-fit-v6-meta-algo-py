# Imputation: porting the federated MICE fix into the meta-algo

This branch brings the corrected federated MICE from
`strata-fit-v6-imputation-py` into `strata-fit-v6-meta-algo-py`.

## 1. Where the fix comes from

Two fix branches existed in `strata-fit-v6-imputation-py`, both built on top of
`testing/adding-site-comparison-test` (Josef's local-vs-federated comparison script):

| | `fix/mice-replay-imputation-sequence` (Chiara) | `fix/change-imputation-loop-and-column-order` (Josef) |
|---|---|---|
| Replays every fitted regression in fitting order (not only the latest estimate per feature) | yes (`imputation_sequence`) | yes (`estimate_sequence`) |
| Visits features in ascending missingness (sklearn `imputation_order="ascending"`) | no, column order | yes (`feature_order`) |
| Final `impute()` replays the sequence once (no extra `max_iter` loop) | yes | yes |
| Legacy models (`global_estimates` only) still work | yes | yes |
| Payload per round | sends both `global_estimates` and the sequence | sends only the sequence; `global_estimates` built at the end |
| Vectorised numpy replay | no (pandas `.loc`) | yes |
| Test against sklearn `IterativeImputer(LinearRegression)` | no | yes |

Running Josef's sklearn-equivalence test against each branch:

| `max_iter` | `main` | Chiara's branch | Josef's branch |
|---|---|---|---|
| 1 | 15.6 % cells differ (max 2.07) | 15.6 % (max 2.07) | match |
| 3 | 13.3 % (max 0.70) | 11.4 % (max 0.42) | match |
| 10 | 13.1 % (max 0.33) | 1.2 % (max 0.25) | match |

The replay fix alone gets close as iterations increase. The visiting order is what
makes it match exactly. **Josef's branch was chosen**: it contains everything
Chiara's does, plus the ordering and the test.

## 2. What was wrong in the meta-algo

`strata_fit_v6_meta_algo_py/imputation.py` carried an older, separate copy of MICE:

* **No iterations at all.** `compute()` built X'X/X'y for *all* columns in one go
  on mean-filled data; `aggregate()` solved them once. That is one regression per
  feature fitted on mean-imputed predictors, not chained equations.
* `impute()` applied each estimate once in column order. Inside the loop it also
  fitted a `BayesianRidge` `IterativeImputer` on the full matrix **and discarded
  the result** (slow, no effect on output).
* `parameters.max_iter` was written into the model but never drove anything.

## 3. Changes on this branch

| File | Change |
|---|---|
| `imputation.py` | `MiceImputer` rewritten after Josef's version: round-0 sums/counts → `initial_means` + `feature_order`; per-feature `compute()` on a matrix rebuilt by `_replay()`; `aggregate()` solves one pooled ridge regression per call; `impute()` replays `estimate_sequence`. New `fit_mice_model(run_partial, columns, max_iter)` holds the central loop and doesn't depend on the transport. Unused `IterativeImputer`/`BayesianRidge` imports removed. Mean/median/constant untouched. |
| `contracts.py` | `ImputationComputePartialInput.global_state: Optional[dict]` (MICE chain state + `target_feat_idx`). `MetaCentralInput.imputation_max_iter: int = 5`. |
| `methods.py` | `imputation_compute_partial_handler` passes `global_state` to `compute()`. `central_handler`: if strategy is MICE, call `fit_mice_model` (with `runner.run` on `imputation_compute_partial`); other strategies keep the single-round path. Downstream steps (linear, Cox, KM, survival bundle) are unchanged: they still receive `global_metrics` = the model config. |
| `partial.py` | `imputation_compute_partial[_frame]` accept and forward `global_state` (also added to `named_arguments`). |
| `central.py` | `main()` accepts and forwards `imputation_max_iter`. |
| `tests/test_mice_imputation.py` | Josef's tests ported (sklearn equivalence for `max_iter` 1/3/10, state shape and order, independence from how rows are split across sites, legacy models), plus an end-to-end central run with `imputation_strategy="mice"` through `InProcessAlgorithmClient`. |

The model config format is a superset of the old one (`global_estimates` is
still there), and `impute()` falls back to it when `estimate_sequence` is missing.

## 4. Points for the team to discuss

1. **Number of tasks.** MICE now runs `1 + imputation_max_iter × len(columns)`
   partial rounds before the final model. With the 7 columns of the
   `mice_imputation_smoke` scenario and the default 5 iterations, that's 36 rounds.
   Check this is acceptable on the real infrastructure (or lower the default).
2. **Fit set vs. apply set.** MICE is fitted on the whole node dataframe. Survival
   flows then impute the cohort-filtered subset (`_filter_and_impute_for_survival`).
   This was already the case before; it's flagged here because it's now more visible.
3. **No convergence criterion.** Like the test with `tol=0`, it always runs
   `max_iter` iterations. sklearn's early stopping is not reproduced.
4. **Deterministic only.** Single imputation by conditional mean (no posterior
   sampling, no multiple imputations). The same limitation as in the imputation repo.
5. Josef's branch moves `pytest` into the runtime dependencies of the imputation
   repo's `pyproject.toml`. It should probably stay under `[dev]`.

## 5. Verification

* `pytest tests --ignore=tests/infra`: 34 passed, 1 failed. The failure is
  `test_client_polling::test_wait_for_task_completion_fails_after_transient_error_budget`,
  which fails on `main` too and is unrelated.
* Run locally on Python 3.13 with numpy 2.x, because `numpy==1.26.0` has no 3.13
  wheel. The pin was not changed. The infra/docker lanes have not been run.
