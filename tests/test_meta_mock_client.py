from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

from strata_fit_v6_meta_algo_py import run_local_meta_algorithm
from tests.synthetic_strata_fit_data import SyntheticConfig, write_partitioned_csv


IMPUTATION_COLUMNS = ["DAS28", "CRP", "HAQ", "Pat_global", "Ph_global", "Pain", "eq5d"]
COX_EXPL_VARS = ["Age_diagnosis", "DAS28", "CRP", "HAQ", "RF_positivity"]


def _load_datasets(tmp_path: Path) -> list[pd.DataFrame]:
    csv_paths = write_partitioned_csv(
        output_dir=tmp_path,
        config=SyntheticConfig(node_count=3, patients_per_node=32, seed=20260318),
    )
    return [pd.read_csv(path) for path in csv_paths]


def _load_small_datasets(tmp_path: Path) -> list[pd.DataFrame]:
    csv_paths = write_partitioned_csv(
        output_dir=tmp_path,
        config=SyntheticConfig(node_count=3, patients_per_node=16, seed=20260318),
    )
    return [pd.read_csv(path) for path in csv_paths]


def test_meta_raw_stratafit_cox_end_to_end(tmp_path: Path) -> None:
    result = run_local_meta_algorithm(
        _load_datasets(tmp_path),
        columns=IMPUTATION_COLUMNS,
        run_validation=True,
        model_name="PatientData",
        imputation_strategy="mean",
        final_model="cox",
        final_model_config={
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": COX_EXPL_VARS,
            "max_iterations": 12,
            "tolerance": 1e-6,
            "preprocess_raw_data": True,
        },
    )

    final_result = result["final_result"]
    assert result["final_model"] == "cox"
    assert len(result["validation"]) == 3
    assert all(item["validation_passed"] for item in result["validation"])
    assert "model" in final_result
    assert final_result["included_organizations"]


def test_meta_raw_stratafit_km_end_to_end(tmp_path: Path) -> None:
    result = run_local_meta_algorithm(
        _load_datasets(tmp_path),
        columns=IMPUTATION_COLUMNS,
        run_validation=True,
        model_name="PatientData",
        imputation_strategy="mean",
        final_model="km",
        final_model_config={
            "preprocess_raw_data": True,
        },
    )

    final_result = result["final_result"]
    assert result["final_model"] == "km"
    assert len(result["validation"]) == 3
    assert all(item["validation_passed"] for item in result["validation"])

    km_curve = pd.read_json(StringIO(final_result["km_curve"]))
    assert not km_curve.empty
    assert "cumulative_incidence" in km_curve.columns


def test_meta_raw_stratafit_survival_bundle_end_to_end(tmp_path: Path) -> None:
    result = run_local_meta_algorithm(
        _load_small_datasets(tmp_path),
        columns=IMPUTATION_COLUMNS,
        run_validation=True,
        model_name="PatientData",
        imputation_strategy="mean",
        final_model="survival_bundle",
        final_model_config={
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": COX_EXPL_VARS,
            "max_iterations": 12,
            "tolerance": 1e-6,
            "preprocess_raw_data": True,
            "event_definition": "d2t_ra_v2026_selected_v1",
            "cohort": {"population": "all_ra"},
            "horizons_months": [12, 24, 60],
            "include_definition_sensitivity": False,
        },
    )

    final_result = result["final_result"]
    assert result["final_model"] == "survival_bundle"
    assert len(result["validation"]) == 3
    assert all(item["validation_passed"] for item in result["validation"])
    assert final_result["definition"]["id"] == "d2t_ra_v2026_selected_v1"
    assert isinstance(final_result["incidence"]["series"], list)
    assert isinstance(final_result["prevalence"]["series"], list)
    assert isinstance(final_result["cox"]["coefficients"], list)
    assert isinstance(final_result["risk_stratification"]["groups"], list)
