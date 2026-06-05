#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from strata_fit_v6_meta_algo_py import run_local_meta_algorithm


IMPUTATION_COLUMNS = ["DAS28", "CRP", "HAQ", "Pat_global", "Ph_global", "Pain", "eq5d"]
COX_EXPL_VARS = ["Age_diagnosis", "DAS28", "CRP", "HAQ", "RF_positivity"]


@dataclass
class ScenarioResult:
    name: str
    status: str
    duration_s: float
    error_summary: str
    checks: dict[str, Any]


SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "baseline_3n_full",
        "node_count": 3,
        "patients_per_node": 36,
        "models": ["sklearn_linear", "cox", "km"],
    },
    {
        "name": "fanout_5n_survival",
        "node_count": 5,
        "patients_per_node": 36,
        "models": ["cox", "km"],
    },
    {
        "name": "fanout_8n_km",
        "node_count": 8,
        "patients_per_node": 24,
        "models": ["km"],
    },
    {
        "name": "fanout_8n_cox",
        "node_count": 8,
        "patients_per_node": 24,
        "models": ["cox"],
    },
]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def lane_fail(artifact_path: Path, started: float, reason: str) -> int:
    payload = {
        "lane_name": "stress_matrix",
        "status": "fail",
        "duration_s": round(time.time() - started, 3),
        "scenarios": [],
        "error_summary": reason,
    }
    artifact_path.write_text(json.dumps(payload, indent=2))
    print(reason, file=sys.stderr)
    return 1


def prepare_datasets(output_dir: Path, *, node_count: int, patients_per_node: int) -> list[pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "tests" / "infra" / "prepare_meta_smoke_data.py"),
            "--output-dir",
            str(output_dir),
            "--node-count",
            str(node_count),
            "--patients-per-node",
            str(patients_per_node),
        ],
        check=True,
    )
    return [
        pd.read_csv(output_dir / f"data_bucket{idx + 1}.csv")
        for idx in range(node_count)
    ]


def run_scenario(name: str, datasets: list[pd.DataFrame], model: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "columns": IMPUTATION_COLUMNS,
        "run_validation": True,
        "model_name": "PatientData",
        "imputation_strategy": "mean",
        "final_model": model,
    }
    if model == "sklearn_linear":
        kwargs["final_model_config"] = {
            "predictors": ["Age_diagnosis", "DAS28", "CRP", "HAQ"],
            "outcome": "RF_positivity",
            "n_local_iterations": 25,
            "model_kwargs": {"solver": "lbfgs"},
        }
    elif model == "cox":
        kwargs["final_model_config"] = {
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": COX_EXPL_VARS,
            "max_iterations": 12,
            "tolerance": 1e-6,
            "preprocess_raw_data": True,
        }
    elif model == "km":
        kwargs["final_model_config"] = {
            "preprocess_raw_data": True,
        }

    result = run_local_meta_algorithm(datasets, **kwargs)
    checks: dict[str, Any] = {
        "scenario": name,
        "final_model": result["final_model"],
        "organization_count": len(result["organizations"]),
        "validation_count": len(result["validation"]),
    }
    if model == "sklearn_linear":
        checks["coef_count"] = len(result["final_result"]["model_attributes"]["coef_"][0])
    elif model == "cox":
        checks["included_organizations"] = len(result["final_result"].get("included_organizations", []))
        checks["has_model"] = "model" in result["final_result"]
    elif model == "km":
        checks["km_curve_length"] = len(pd.read_json(StringIO(result["final_result"]["km_curve"])))
    return checks


def run() -> int:
    started = time.time()
    run_id = os.getenv("RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    artifact_dir = Path(os.getenv("ARTIFACT_DIR", str(ROOT_DIR / "ARTIFACTS" / "validation")))
    scenario_artifacts = artifact_dir / "scenarios"
    scenario_artifacts.mkdir(parents=True, exist_ok=True)
    lane_artifact = artifact_dir / "stress_matrix_lane.json"

    results: list[ScenarioResult] = []
    lane_errors: list[str] = []
    tmp_root = Path("/tmp") / f"strata-meta-stress-{run_id}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    for scenario in SCENARIOS:
        s_start = time.time()
        name = scenario["name"]
        try:
            datasets = prepare_datasets(
                tmp_root / name,
                node_count=scenario["node_count"],
                patients_per_node=scenario["patients_per_node"],
            )
            model_checks = []
            for model in scenario["models"]:
                model_checks.append(run_scenario(name, datasets, model))
            checks = {
                "node_count": scenario["node_count"],
                "patients_per_node": scenario["patients_per_node"],
                "models": model_checks,
            }
            artifact = {
                "run_id": run_id,
                "timestamp_utc": now_utc(),
                "scenario": name,
                "status": "pass",
                "duration_s": round(time.time() - s_start, 3),
                "checks": checks,
            }
            (scenario_artifacts / f"{name}.json").write_text(json.dumps(artifact, indent=2))
            results.append(
                ScenarioResult(
                    name=name,
                    status="pass",
                    duration_s=artifact["duration_s"],
                    error_summary="",
                    checks=checks,
                )
            )
        except Exception as exc:
            duration = round(time.time() - s_start, 3)
            lane_errors.append(f"{name}: {exc}")
            fail_artifact = {
                "run_id": run_id,
                "timestamp_utc": now_utc(),
                "scenario": name,
                "status": "fail",
                "duration_s": duration,
                "checks": {},
                "error_summary": str(exc),
            }
            (scenario_artifacts / f"{name}.json").write_text(json.dumps(fail_artifact, indent=2))
            results.append(
                ScenarioResult(name=name, status="fail", duration_s=duration, error_summary=str(exc), checks={})
            )

    lane_status = "pass" if not lane_errors else "fail"
    lane_artifact.write_text(
        json.dumps(
            {
                "lane_name": "stress_matrix",
                "status": lane_status,
                "duration_s": round(time.time() - started, 3),
                "scenarios": [result.__dict__ for result in results],
                "error_summary": "; ".join(lane_errors),
            },
            indent=2,
        )
    )
    return 0 if lane_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(run())
