from pathlib import Path

import pandas as pd
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


def _build_client_with_uniform_data(tmp_path: Path) -> MockAlgorithmClient:
    paths = []
    for idx in range(3):
        df = pd.DataFrame(
            {
                "pat_ID": range(1, 16),
                "time": [float(x + 1) for x in range(15)],
                "event": [1 if x % 2 == 0 else 0 for x in range(15)],
                "x1": [float((x % 5) + idx) if x % 4 else None for x in range(15)],
                "x2": [float((x % 7) + 1) for x in range(15)],
                "outcome": [1 if x % 3 == 0 else 0 for x in range(15)],
            }
        )
        path = tmp_path / f"uniform_{idx}.csv"
        df.to_csv(path, index=False)
        paths.append(path)

    datasets = [[{"database": path, "db_type": "csv"}] for path in paths]
    return MockAlgorithmClient(datasets=datasets, module="strata_fit_v6_meta_algo_py")


def test_legacy_sklearn_linear_arguments_supported(tmp_path: Path) -> None:
    client = _build_client_with_uniform_data(tmp_path)
    org_ids = [organization["id"] for organization in client.organization.list()]

    task = client.task.create(
        input_={
            "master": True,
            "method": "main",
            "kwargs": {
                "columns": ["x1", "x2"],
                "predictors": ["x1", "x2"],
                "outcome": "outcome",
                "n_local_iterations": 12,
                "run_validation": False,
                "imputation_strategy": "mean",
                "organizations": org_ids,
            },
        },
        organizations=[org_ids[0]],
    )
    result = client.result.get(task["id"])
    assert result["final_model"] == "sklearn_linear"
    assert result["final_result"]["model_attributes"]["coef_"]


def test_validation_results_are_returned(tmp_path: Path) -> None:
    client = _build_client_with_uniform_data(tmp_path)
    org_ids = [organization["id"] for organization in client.organization.list()]

    task = client.task.create(
        input_={
            "master": True,
            "method": "main",
            "kwargs": {
                "columns": ["x1", "x2"],
                "run_validation": True,
                "model_name": "PatientData",
                "imputation_strategy": "mean",
                "final_model": "sklearn_linear",
                "organizations": org_ids,
                "final_model_config": {
                    "predictors": ["x1", "x2"],
                    "outcome": "outcome",
                    "n_local_iterations": 10,
                },
            },
        },
        organizations=[org_ids[0]],
    )
    result = client.result.get(task["id"])
    assert len(result["validation"]) == len(org_ids)
    assert "validation_passed" in result["validation"][0]


def test_cox_handles_low_event_threshold(tmp_path: Path) -> None:
    paths = []
    for idx in range(2):
        df = pd.DataFrame(
            {
                "pat_ID": range(1, 8),
                "time": [float(x + 1) for x in range(7)],
                "event": [0 for _ in range(7)],
                "x1": [float(x + idx) for x in range(7)],
                "x2": [float(x + 2) for x in range(7)],
            }
        )
        path = tmp_path / f"low_event_{idx}.csv"
        df.to_csv(path, index=False)
        paths.append(path)

    datasets = [[{"database": path, "db_type": "csv"}] for path in paths]
    client = MockAlgorithmClient(datasets=datasets, module="strata_fit_v6_meta_algo_py")
    org_ids = [organization["id"] for organization in client.organization.list()]

    task = client.task.create(
        input_={
            "master": True,
            "method": "main",
            "kwargs": {
                "columns": ["x1", "x2"],
                "run_validation": False,
                "imputation_strategy": "mean",
                "final_model": "cox",
                "organizations": org_ids,
                "final_model_config": {
                    "time_col": "time",
                    "outcome_col": "event",
                    "expl_vars": ["x1", "x2"],
                },
            },
        },
        organizations=[org_ids[0]],
    )
    result = client.result.get(task["id"])
    cox = result["final_result"]
    assert cox["included_organizations"] == []
    assert "minimum event threshold" in cox["warnings"][0].lower()
