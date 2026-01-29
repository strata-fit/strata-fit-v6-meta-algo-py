from pathlib import Path
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


def build_client():
    data_dir = Path("v6-infra/infrastructure/data/meta")
    datasets = [
        [{"database": data_dir / "alpha.csv", "db_type": "csv"}],
        [{"database": data_dir / "beta.csv", "db_type": "csv"}],
        [{"database": data_dir / "gamma.csv", "db_type": "csv"}],
    ]
    return MockAlgorithmClient(datasets=datasets, module="strata_fit_v6_meta_algo_py")


def test_meta_algorithm_end_to_end():
    client = build_client()
    org_ids = [o["id"] for o in client.organization.list()]

    task = client.task.create(
        input_={
            "master": True,
            "method": "main",
            "kwargs": {
                "columns": ["DAS28", "CRP", "HAQ", "Pat_global", "Pain"],
                "predictors": ["Age_diagnosis", "DAS28", "CRP", "HAQ"],
                "outcome": "RF_positivity",
                "n_local_iterations": 30,
                "run_validation": True,
                "organizations": org_ids,
            },
        },
        organizations=[org_ids[0]],
    )

    result = client.result.get(task["id"])

    assert set(
        ["imputation_metrics", "lr_partials", "lr_model_attributes", "organizations"]
    ).issubset(result.keys())

    assert result["lr_model_attributes"].get("coef_"), "Coefficients missing"

    # Ensure each partial trained on some rows (imputation should remove NaNs in predictors)
    for partial in result["lr_partials"]:
        assert partial.get("size", 0) > 0
