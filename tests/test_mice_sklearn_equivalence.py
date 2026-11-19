"""Ported from strata-fit-v6-imputation-py test/test_mice_sklearn_equivalence.py.

Lines marked META-ALGO differ from the original.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# META-ALGO: imports from this repo
from strata_fit_v6_meta_algo_py.central import main
from strata_fit_v6_meta_algo_py.imputation import MiceImputer, fit_mice_model
from strata_fit_v6_meta_algo_py.local_client import InProcessAlgorithmClient

sklearn = pytest.importorskip("sklearn")
from sklearn.experimental import enable_iterative_imputer  # noqa: E402,F401
from sklearn.impute import IterativeImputer  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

COLUMNS = ["A", "B", "C", "D", "E"]


def _split(frame: pd.DataFrame, n_sites: int):
    return [
        frame.iloc[idx].reset_index(drop=True)
        for idx in np.array_split(np.arange(len(frame)), n_sites)
    ]


def _make_sites(seed: int = 0, n: int = 450, n_sites: int = 3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    data = np.column_stack(
        [
            2 * z[:, 0] + rng.normal(size=n),
            z[:, 0] + z[:, 1] + 0.3 * rng.normal(size=n),
            3 * z[:, 1] + rng.normal(size=n),
            z[:, 0] - z[:, 1] + 0.5 * rng.normal(size=n),
            rng.normal(size=n) + z[:, 1],
        ]
    )
    # different missingness per column so the imputation order is non-trivial
    for j, frac in enumerate([0.30, 0.05, 0.20, 0.10, 0.15]):
        data[rng.random(n) < frac, j] = np.nan
    frame = pd.DataFrame(data, columns=COLUMNS)
    return frame, _split(frame, n_sites)


def _fit(sites, max_iter):
    # META-ALGO: fit through fit_mice_model, nodes simulated in-process
    imputer = MiceImputer()
    return fit_mice_model(
        lambda state: [imputer.compute(site, COLUMNS, global_state=state) for site in sites],
        columns=COLUMNS,
        max_iter=max_iter,
    )


@pytest.mark.parametrize("max_iter", [1, 3, 10])
def test_federated_mice_matches_sklearn_ols(max_iter: int) -> None:
    pooled, sites = _make_sites()
    model = _fit(sites, max_iter)

    imputer = MiceImputer()
    federated = pd.concat([imputer.impute(site.copy(), model) for site in sites])[COLUMNS]

    local = IterativeImputer(
        estimator=LinearRegression(), max_iter=max_iter, tol=0.0, random_state=0
    ).fit_transform(pooled.to_numpy())

    np.testing.assert_allclose(federated.to_numpy(), local, atol=1e-3, rtol=1e-3)


def test_state_contains_full_sequence_and_order() -> None:
    _, sites = _make_sites()
    model = _fit(sites, 4)
    state = model["state"]

    assert len(state["estimate_sequence"]) == 4 * len(COLUMNS)
    assert sorted(state["feature_order"]) == list(range(len(COLUMNS)))
    # ascending missingness: B (5%) first, A (30%) last
    assert state["feature_order"][0] == 1 and state["feature_order"][-1] == 0
    assert len(state["global_estimates"]) == len(COLUMNS)


def test_result_is_independent_of_the_site_split() -> None:
    pooled, sites3 = _make_sites(n_sites=3)
    sites4 = _split(pooled, 4)
    imputer = MiceImputer()
    out3 = pd.concat([imputer.impute(s.copy(), _fit(sites3, 3)) for s in sites3])[COLUMNS]
    out4 = pd.concat([imputer.impute(s.copy(), _fit(sites4, 3)) for s in sites4])[COLUMNS]
    np.testing.assert_allclose(out3.to_numpy(), out4.to_numpy(), atol=1e-8)


def test_legacy_models_without_sequence_still_impute() -> None:
    _, sites = _make_sites()
    model = _fit(sites, 3)
    legacy = {**model, "state": {
        "initial_means": model["state"]["initial_means"],
        "global_estimates": model["state"]["global_estimates"],
    }}
    out = MiceImputer().impute(sites[0].copy(), legacy)
    assert not out[COLUMNS].isna().any().any()


# META-ALGO: additional test - full central flow with MICE through the in-process client
def test_central_flow_runs_iterative_mice_end_to_end() -> None:
    _, sites = _make_sites(n=150)
    for site in sites:
        site["pat_ID"] = range(len(site))
        site["outcome"] = (site["B"].fillna(0.0) > 0).astype(int)
    result = main(
        client=InProcessAlgorithmClient(sites),
        columns=COLUMNS,
        run_validation=False,
        imputation_strategy="mice",
        imputation_max_iter=2,
        final_model="sklearn_linear",
        final_model_config={"predictors": COLUMNS, "outcome": "outcome", "n_local_iterations": 4},
    )
    state = result["imputation_metrics"]["state"]
    assert len(state["estimate_sequence"]) == 2 * len(COLUMNS)
    assert result["imputation_metrics"]["parameters"]["max_iter"] == 2
    assert result["final_result"]["partials"]
