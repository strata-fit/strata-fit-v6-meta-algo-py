from __future__ import annotations

from pathlib import Path
from io import StringIO

import pandas as pd
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

from tests.synthetic_strata_fit_data import SyntheticConfig, write_partitioned_csv


IMPUTATION_COLUMNS = ["DAS28", "CRP", "HAQ", "Pat_global", "Ph_global", "Pain", "eq5d"]
COX_EXPL_VARS = ["Age_diagnosis", "DAS28", "CRP", "HAQ", "RF_positivity"]


def _build_client_from_paths(paths: list[Path]) -> MockAlgorithmClient:
    datasets = [[{"database": path, "db_type": "csv"}] for path in paths]
    return MockAlgorithmClient(datasets=datasets, module="strata_fit_v6_meta_algo_py")


def _build_client_with_stratafit_data(tmp_path: Path) -> tuple[MockAlgorithmClient, list[Path]]:
    csv_paths = write_partitioned_csv(
        output_dir=tmp_path,
        config=SyntheticConfig(node_count=3, patients_per_node=32, seed=20260318),
    )
    return _build_client_from_paths(csv_paths), csv_paths


def test_meta_raw_stratafit_cox_end_to_end(tmp_path: Path) -> None:
    client, _ = _build_client_with_stratafit_data(tmp_path)
    org_ids = [organization["id"] for organization in client.organization.list()]

    task = client.task.create(
        input_={
            "master": True,
            "method": "main",
            "kwargs": {
                "columns": IMPUTATION_COLUMNS,
                "run_validation": True,
                "model_name": "PatientData",
                "imputation_strategy": "mean",
                "final_model": "cox",
                "organizations": org_ids,
                "final_model_config": {
                    "time_col": "time",
                    "outcome_col": "event",
                    "expl_vars": COX_EXPL_VARS,
                    "max_iterations": 12,
                    "tolerance": 1e-6,
                    "preprocess_raw_data": True,
                },
            },
        },
        organizations=[org_ids[0]],
    )

    result = client.result.get(task["id"])
    final_result = result["final_result"]

    assert result["final_model"] == "cox"
    assert len(result["validation"]) == len(org_ids)
    assert all(item["validation_passed"] for item in result["validation"])
    assert "model" in final_result
    assert final_result["included_organizations"]


def test_meta_raw_stratafit_km_end_to_end(tmp_path: Path) -> None:
    client, _ = _build_client_with_stratafit_data(tmp_path)
    org_ids = [organization["id"] for organization in client.organization.list()]

    task = client.task.create(
        input_={
            "master": True,
            "method": "main",
            "kwargs": {
                "columns": IMPUTATION_COLUMNS,
                "run_validation": True,
                "model_name": "PatientData",
                "imputation_strategy": "mean",
                "final_model": "km",
                "organizations": org_ids,
                "final_model_config": {
                    "preprocess_raw_data": True,
                },
            },
        },
        organizations=[org_ids[0]],
    )

    result = client.result.get(task["id"])
    final_result = result["final_result"]

    assert result["final_model"] == "km"
    assert len(result["validation"]) == len(org_ids)
    assert all(item["validation_passed"] for item in result["validation"])

    km_curve = pd.read_json(StringIO(final_result["km_curve"]))
    assert not km_curve.empty
    assert "cumulative_incidence" in km_curve.columns
