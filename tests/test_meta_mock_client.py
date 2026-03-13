from pathlib import Path

import pandas as pd
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


def _build_client_from_paths(paths: list[Path]) -> MockAlgorithmClient:
    datasets = [[{"database": path, "db_type": "csv"}] for path in paths]
    return MockAlgorithmClient(datasets=datasets, module="strata_fit_v6_meta_algo_py")


def test_meta_sklearn_linear_end_to_end(tmp_path: Path) -> None:
    csv_paths: list[Path] = []
    for idx in range(3):
        df = pd.DataFrame(
            {
                "pat_ID": [f"node{idx}_row{row}" for row in range(1, 26)],
                "Age_diagnosis": [35 + idx + (row % 20) for row in range(25)],
                "DAS28": [2.5 + (row % 6) * 0.3 if row % 7 else None for row in range(25)],
                "CRP": [1.2 + (row % 5) * 0.8 if row % 8 else None for row in range(25)],
                "HAQ": [0.4 + (row % 4) * 0.2 if row % 9 else None for row in range(25)],
                "Pat_global": [35 + (row % 30) for row in range(25)],
                "Pain": [30 + ((row + idx) % 35) for row in range(25)],
                "RF_positivity": [1 if (row + idx) % 2 == 0 else 0 for row in range(25)],
            }
        )
        path = tmp_path / f"linear_node_{idx}.csv"
        df.to_csv(path, index=False)
        csv_paths.append(path)

    client = _build_client_from_paths(csv_paths)
    org_ids = [organization["id"] for organization in client.organization.list()]

    task = client.task.create(
        input_={
            "master": True,
            "method": "main",
            "kwargs": {
                "columns": ["DAS28", "CRP", "HAQ", "Pat_global", "Pain"],
                "run_validation": False,
                "imputation_strategy": "mean",
                "final_model": "sklearn_linear",
                "organizations": org_ids,
                "final_model_config": {
                    "predictors": ["Age_diagnosis", "DAS28", "CRP", "HAQ"],
                    "outcome": "RF_positivity",
                    "n_local_iterations": 15,
                },
            },
        },
        organizations=[org_ids[0]],
    )

    result = client.result.get(task["id"])
    assert result["final_model"] == "sklearn_linear"
    assert set(["final_result", "imputation_metrics", "organizations"]).issubset(result.keys())
    assert len(result["final_result"]["partials"]) == len(org_ids)
    assert "coef_" in result["final_result"]["model_attributes"]


def test_meta_cox_end_to_end(tmp_path: Path) -> None:
    csv_paths: list[Path] = []
    for idx in range(3):
        df = pd.DataFrame(
            {
                "pat_ID": range(1, 31),
                "time": [float(x + 1) for x in range(30)],
                "event": [1 if x % 2 == 0 else 0 for x in range(30)],
                "x1": [float((x % 7) + idx) if x % 5 else None for x in range(30)],
                "x2": [float((x % 11) + 2) for x in range(30)],
            }
        )
        path = tmp_path / f"node_{idx}.csv"
        df.to_csv(path, index=False)
        csv_paths.append(path)

    client = _build_client_from_paths(csv_paths)
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
                    "max_iterations": 8,
                    "tolerance": 1e-6,
                },
            },
        },
        organizations=[org_ids[0]],
    )

    result = client.result.get(task["id"])
    final_result = result["final_result"]
    assert result["final_model"] == "cox"
    assert "model" in final_result
    assert final_result["included_organizations"]
