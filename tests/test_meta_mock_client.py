import os
import importlib.util
from pathlib import Path
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


def _ensure_validator_config_path():
    """Point CONFIG_PATH env var to the installed schema config so Dynaconf can load it."""
    if os.environ.get("CONFIG_PATH"):
        return
    spec = importlib.util.find_spec("strata_fit_v6_data_validator_py")
    if not spec or not spec.origin:
        return
    config_dir = Path(spec.origin).parent.parent / "config"
    if config_dir.exists():
        os.environ["CONFIG_PATH"] = str(config_dir)


def build_client():
    _ensure_validator_config_path()
    data_dir = Path("v6-infra/infrastructure/data/meta")
    datasets = [
        [{"database": data_dir / "alpha.csv", "db_type": "csv"}],
        [{"database": data_dir / "beta.csv", "db_type": "csv"}],
        [{"database": data_dir / "gamma.csv", "db_type": "csv"}],
    ]
    return MockAlgorithmClient(
        datasets=datasets, module="strata_fit_v6_meta_algo_py.central"
    )


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
    print(result)

    assert set(
        ["imputation_metrics", "lr_partials", "lr_model_attributes", "organizations"]
    ).issubset(result.keys())

    assert result["lr_model_attributes"].get("coef_"), "Coefficients missing"

    # Ensure each partial trained on some rows (imputation should remove NaNs in predictors)
    for partial in result["lr_partials"]:
        assert partial.get("size", 0) > 0

if __name__ == "__main__":
    test_meta_algorithm_end_to_end()
