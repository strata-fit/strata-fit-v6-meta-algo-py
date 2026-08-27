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


def test_survival_bundle_contains_incidence_prevalence_cox_and_risk_groups() -> None:
    datasets = []
    for idx in range(3):
        datasets.append(
            pd.DataFrame(
                {
                    "pat_ID": [f"P{idx}_1" for _ in range(4)] + [f"P{idx}_2" for _ in range(4)],
                    "Visit_months_from_diagnosis": [0.0, 6.0, 12.0, 24.0] * 2,
                    "Age_diagnosis": [50] * 4 + [61] * 4,
                    "Sex": [1] * 4 + [0] * 4,
                    "RF_positivity": [1] * 8,
                    "anti_CCP": [1] * 8,
                    "DAS28": [2.1, 3.3, 3.5, 3.4, 2.0, 2.3, 2.5, 2.4],
                    "CRP": [6.0, 9.0, 11.0, 11.0, 5.0, 5.0, 6.0, 6.0],
                    "Pat_global": [20.0, 60.0, 60.0, 60.0, 20.0, 20.0, 20.0, 20.0],
                    "Ph_global": [20.0] * 8,
                    "Pain": [45.0] * 8,
                    "HAQ": [1.0] * 8,
                    "eq5d": [0.8] * 8,
                    "csDMARD1": [1] * 8,
                    "N_prev_csDMARD": [1] * 8,
                    "bDMARD": [1, 2, 2, 2, 1, 2, 2, 2],
                    "N_prev_bDMARD": [0] * 8,
                    "tsDMARD": [None, 1, 1, 1, None, 1, 1, 1],
                    "N_prev_tsDMARD": [0] * 8,
                    "GC": [0] * 8,
                    "GC_type": [1] * 8,
                    "Year_diagnosis": [2012] * 8,
                    "month_diagnosis": [1] * 8,
                }
            )
        )

    result = run_local_meta_algorithm(
        datasets,
        columns=["DAS28", "CRP", "HAQ", "Pat_global", "Ph_global", "Pain", "eq5d"],
        run_validation=False,
        imputation_strategy="mean",
        final_model="survival_bundle",
        final_model_config={
            "preprocess_raw_data": True,
            "max_iterations": 3,
            "cohort": {"population": "all_ra"},
            "event_definition": "d2t_ra_v2026_selected_v1",
            "horizons_months": [12, 24, 60],
        },
    )

    final_result = result["final_result"]
    assert result["final_model"] == "survival_bundle"
    assert {"definition", "incidence", "prevalence", "cox", "risk_stratification", "validation", "imputation", "metadata"} <= set(final_result)
    assert isinstance(final_result["incidence"]["series"], list)
    assert isinstance(final_result["prevalence"]["series"], list)
    assert isinstance(final_result["cox"]["coefficients"], list)
    assert isinstance(final_result["risk_stratification"]["groups"], list)
