from __future__ import annotations

import pandas as pd

from strata_fit_v6_meta_algo_py import run_local_meta_algorithm


def _build_uniform_datasets() -> list[pd.DataFrame]:
    datasets: list[pd.DataFrame] = []
    for idx in range(3):
        datasets.append(
            pd.DataFrame(
                {
                    "pat_ID": range(1, 16),
                    "time": [float(x + 1) for x in range(15)],
                    "event": [1 if x % 2 == 0 else 0 for x in range(15)],
                    "x1": [float((x % 5) + idx) if x % 4 else None for x in range(15)],
                    "x2": [float((x % 7) + 1) for x in range(15)],
                    "outcome": [1 if x % 3 == 0 else 0 for x in range(15)],
                }
            )
        )
    return datasets


def test_legacy_sklearn_linear_arguments_supported() -> None:
    result = run_local_meta_algorithm(
        _build_uniform_datasets(),
        columns=["x1", "x2"],
        predictors=["x1", "x2"],
        outcome="outcome",
        n_local_iterations=12,
        run_validation=False,
        imputation_strategy="mean",
    )
    assert result["final_model"] == "sklearn_linear"
    assert result["final_result"]["model_attributes"]["coef_"]


def test_validation_results_are_returned() -> None:
    result = run_local_meta_algorithm(
        _build_uniform_datasets(),
        columns=["x1", "x2"],
        run_validation=True,
        model_name="PatientData",
        imputation_strategy="mean",
        final_model="sklearn_linear",
        final_model_config={
            "predictors": ["x1", "x2"],
            "outcome": "outcome",
            "n_local_iterations": 10,
        },
    )
    assert len(result["validation"]) == 3
    assert "validation_passed" in result["validation"][0]


def test_cox_handles_low_event_threshold() -> None:
    datasets = []
    for idx in range(2):
        datasets.append(
            pd.DataFrame(
                {
                    "pat_ID": range(1, 8),
                    "time": [float(x + 1) for x in range(7)],
                    "event": [0 for _ in range(7)],
                    "x1": [float(x + idx) for x in range(7)],
                    "x2": [float(x + 2) for x in range(7)],
                }
            )
        )

    result = run_local_meta_algorithm(
        datasets,
        columns=["x1", "x2"],
        run_validation=False,
        imputation_strategy="mean",
        final_model="cox",
        final_model_config={
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": ["x1", "x2"],
        },
    )
    cox = result["final_result"]
    assert cox["included_organizations"] == []
    assert "minimum event threshold" in cox["warnings"][0].lower()
