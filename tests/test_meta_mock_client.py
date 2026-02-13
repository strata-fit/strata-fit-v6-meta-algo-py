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
    data_path = Path("/Users/cripepi2/Documents/data_11022026.csv")
    datasets = [
        [{"database": data_path, "db_type": "csv"}],
    ]

    return MockAlgorithmClient(
        datasets=datasets, module="strata_fit_v6_meta_algo_py.central"
    )

def test_meta_algorithm_end_to_end():
    client = build_client()
    org_ids = [o["id"] for o in client.organization.list()]
    assert len(org_ids) == 1, f"Expected 1 org, got {len(org_ids)}"

    # IMPORTANT: call the CENTRAL entrypoint
    task = client.task.create(
        input_={
            "master": True,
            "method": "main",
            "kwargs": {
                "columns": ["DAS28", "CRP", "Pat_global", "Pain"],
                "predictors": ["Year_diagnosis",  "cum_btsDMARDmin"],
                "outcome": "D2T_RA_Ever", ## Still LR label for testing; KM will use D2T-event columns
                "n_local_iterations": 30,
                "run_validation": True,
                "organizations": org_ids,

                # NEW: KM ON
                "run_km": True,
                # optional noise params; omit or leave None if you want defaults
                "km_noise_type": None,
                "km_snr": None,
                "km_random_seed": None,
            },
        },
        organizations=org_ids,
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

    assert "km_pooled" in result, "Missing pooled KM output"
    assert "meta_outcome_d2t_events" in result, "Missing meta D2T outcome"
    assert result["meta_outcome_d2t_events"] is not None, "D2T outcome is None"


if __name__ == "__main__":
    test_meta_algorithm_end_to_end()