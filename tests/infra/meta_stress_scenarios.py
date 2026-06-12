from __future__ import annotations

from copy import deepcopy
from typing import Any


IMPUTATION_COLUMNS = ["DAS28", "CRP", "HAQ", "Pat_global", "Ph_global", "Pain", "eq5d"]
COX_EXPL_VARS = ["Age_diagnosis", "DAS28", "CRP", "HAQ", "RF_positivity"]
DERIVED_PREDICTOR_COLUMNS = {
    "D2T_RA",
    "D2T_RA_Ever",
    "D2T_crit1",
    "D2T_crit2",
    "D2T_crit3",
    "TTE",
    "cens",
    "event_type",
    "interval_start",
    "interval_end",
    "time",
    "event",
}


def _linear_config(predictors: list[str]) -> dict[str, Any]:
    return {
        "predictors": predictors,
        "outcome": "RF_positivity",
        "n_local_iterations": 25,
        "model_kwargs": {"solver": "lbfgs"},
    }


def _cox_config(expl_vars: list[str]) -> dict[str, Any]:
    return {
        "time_col": "time",
        "outcome_col": "event",
        "expl_vars": expl_vars,
        "max_iterations": 12,
        "tolerance": 1e-6,
        "preprocess_raw_data": True,
    }


def _km_config() -> dict[str, Any]:
    return {"preprocess_raw_data": True}


def _scenario(
    *,
    name: str,
    node_count: int,
    patients_per_node: int,
    models: list[str],
    seed: int,
    imputation_strategy: str = "mean",
    imputation_columns: list[str] | None = None,
    missingness_profile: str = "baseline",
    event_profile: str = "adequate",
    signal_profile: str = "strong",
    site_heterogeneity: bool = False,
    model_configs: dict[str, dict[str, Any]] | None = None,
    infra: bool = False,
    expected: str = "positive",
) -> dict[str, Any]:
    configs = {
        "sklearn_linear": _linear_config(["Age_diagnosis", "DAS28", "CRP", "HAQ"]),
        "cox": _cox_config(COX_EXPL_VARS),
        "km": _km_config(),
    }
    if model_configs:
        configs.update(model_configs)
    return {
        "name": name,
        "node_count": node_count,
        "patients_per_node": patients_per_node,
        "seed": seed,
        "models": models,
        "imputation_strategy": imputation_strategy,
        "imputation_columns": imputation_columns or IMPUTATION_COLUMNS,
        "data_profile": {
            "missingness_profile": missingness_profile,
            "event_profile": event_profile,
            "signal_profile": signal_profile,
            "site_heterogeneity": site_heterogeneity,
        },
        "model_configs": configs,
        "infra": infra,
        "expected": expected,
    }


SCENARIOS: list[dict[str, Any]] = [
    _scenario(
        name="d2t_survival_baseline",
        node_count=3,
        patients_per_node=36,
        models=["cox", "km"],
        seed=20260318,
    ),
    _scenario(
        name="dashboard_full_baseline",
        node_count=3,
        patients_per_node=36,
        models=["sklearn_linear", "cox", "km"],
        seed=20260318,
        infra=True,
    ),
    _scenario(
        name="d2t_criteria_truth_table",
        node_count=1,
        patients_per_node=8,
        models=[],
        seed=20260318,
        expected="correctness",
    ),
    _scenario(
        name="patient_reported_burden",
        node_count=3,
        patients_per_node=36,
        models=["sklearn_linear"],
        seed=20260319,
        imputation_strategy="median",
        model_configs={
            "sklearn_linear": _linear_config(
                ["Pat_global", "Pain", "Ph_global", "HAQ", "DAS28", "eq5d"]
            )
        },
    ),
    _scenario(
        name="lab_inflammation_signal",
        node_count=3,
        patients_per_node=36,
        models=["cox"],
        seed=20260320,
        missingness_profile="lab",
        model_configs={
            "cox": _cox_config(
                ["CRP", "ESR", "SJC28", "TJC28", "DAS28", "Age_diagnosis", "Sex", "RF_positivity"]
            )
        },
    ),
    _scenario(
        name="treatment_history_d2t",
        node_count=3,
        patients_per_node=36,
        models=["cox"],
        seed=20260321,
        imputation_columns=[
            *IMPUTATION_COLUMNS,
            "N_prev_csDMARD",
            "N_prev_bDMARD",
            "N_prev_tsDMARD",
            "bDMARD",
            "tsDMARD",
            "GC",
            "GC_dose",
        ],
        model_configs={
            "cox": _cox_config(
                [
                    "N_prev_csDMARD",
                    "N_prev_bDMARD",
                    "N_prev_tsDMARD",
                    "bDMARD",
                    "tsDMARD",
                    "GC",
                    "GC_dose",
                    "DAS28",
                    "CRP",
                    "HAQ",
                ]
            )
        },
    ),
    _scenario(
        name="site_heterogeneity_5n",
        node_count=5,
        patients_per_node=36,
        models=["cox", "km"],
        seed=20260322,
        site_heterogeneity=True,
        signal_profile="modest",
        infra=True,
    ),
    _scenario(
        name="fanout_8n_km",
        node_count=8,
        patients_per_node=24,
        models=["km"],
        seed=20260323,
        site_heterogeneity=True,
        infra=True,
    ),
    _scenario(
        name="fanout_8n_cox",
        node_count=8,
        patients_per_node=24,
        models=["cox"],
        seed=20260324,
        site_heterogeneity=True,
        infra=True,
    ),
    _scenario(
        name="high_missingness",
        node_count=3,
        patients_per_node=40,
        models=["sklearn_linear", "cox", "km"],
        seed=20260325,
        missingness_profile="high",
    ),
    _scenario(
        name="low_event_edge",
        node_count=3,
        patients_per_node=24,
        models=["cox", "km"],
        seed=20260326,
        event_profile="low",
        expected="edge",
    ),
    _scenario(
        name="null_signal_negative_control",
        node_count=3,
        patients_per_node=36,
        models=["sklearn_linear", "cox", "km"],
        seed=20260327,
        signal_profile="null",
        expected="edge",
    ),
    _scenario(
        name="mice_imputation_smoke",
        node_count=3,
        patients_per_node=18,
        models=["sklearn_linear"],
        seed=20260328,
        imputation_strategy="mice",
    ),
]

SCENARIOS_BY_NAME = {scenario["name"]: scenario for scenario in SCENARIOS}
INFRA_SCENARIO_NAMES = [scenario["name"] for scenario in SCENARIOS if scenario["infra"]]


def get_scenario(name: str) -> dict[str, Any]:
    try:
        return deepcopy(SCENARIOS_BY_NAME[name])
    except KeyError as exc:
        known = ", ".join(sorted(SCENARIOS_BY_NAME))
        raise ValueError(f"Unknown meta stress scenario '{name}'. Known scenarios: {known}") from exc


def validate_no_derived_predictors(config: dict[str, Any]) -> None:
    violations: list[str] = []
    for model_name, model_config in config.get("model_configs", {}).items():
        fields: list[str] = []
        fields.extend(model_config.get("predictors") or [])
        fields.extend(model_config.get("expl_vars") or [])
        for field in fields:
            if field in DERIVED_PREDICTOR_COLUMNS:
                violations.append(f"{model_name}:{field}")
    if violations:
        joined = ", ".join(violations)
        raise ValueError(f"Scenario uses derived D2T/event columns as raw predictors: {joined}")
