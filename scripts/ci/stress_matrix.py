#!/usr/bin/env python3
from __future__ import annotations

import json
import math
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
from tests.infra.meta_stress_scenarios import (
    SCENARIOS,
    validate_no_derived_predictors,
)


@dataclass
class ScenarioResult:
    name: str
    status: str
    duration_s: float
    error_summary: str
    checks: dict[str, Any]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def prepare_datasets(output_dir: Path, scenario: dict[str, Any]) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "tests" / "infra" / "prepare_meta_smoke_data.py"),
            "--output-dir",
            str(output_dir),
            "--scenario",
            scenario["name"],
            "--manifest-path",
            str(manifest_path),
        ],
        check=True,
    )
    datasets = [
        pd.read_csv(output_dir / f"data_bucket{idx + 1}.csv")
        for idx in range(int(scenario["node_count"]))
    ]
    return datasets, json.loads(manifest_path.read_text(encoding="utf-8"))


def _assert_json_safe(payload: Any) -> None:
    json.dumps(_json_safe(payload))


def _assert_d2t_summary(manifest: dict[str, Any]) -> None:
    summaries = [manifest.get("overall", {}).get("d2t_criteria", {})]
    summaries.extend(node.get("d2t_criteria", {}) for node in manifest.get("nodes", []))
    for summary in summaries:
        if not summary:
            raise RuntimeError("missing D2T criteria summary")
        if summary.get("all_three_required") is not True:
            raise RuntimeError("D2T summary indicates events not equal to crit1 & crit2 & crit3")


def run_model(
    *,
    name: str,
    datasets: list[pd.DataFrame],
    scenario: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    result = run_local_meta_algorithm(
        datasets,
        columns=scenario["imputation_columns"],
        run_validation=True,
        model_name="PatientData",
        imputation_strategy=scenario["imputation_strategy"],
        final_model=model,
        final_model_config=scenario["model_configs"][model],
    )
    _assert_json_safe(result)

    checks: dict[str, Any] = {
        "scenario": name,
        "final_model": result["final_model"],
        "organization_count": len(result["organizations"]),
        "validation_count": len(result["validation"]),
        "validations_passed": all(item.get("validation_passed") for item in result["validation"]),
    }
    if model == "sklearn_linear":
        checks["coef_count"] = len(result["final_result"]["model_attributes"]["coef_"][0])
    elif model == "cox":
        checks["included_organizations"] = len(result["final_result"].get("included_organizations", []))
        checks["has_model"] = "model" in result["final_result"]
        checks["warnings"] = result["final_result"].get("warnings", [])
    elif model == "km":
        curve = pd.read_json(StringIO(result["final_result"]["km_curve"]))
        checks["km_curve_length"] = len(curve)
        checks["has_cumulative_incidence"] = "cumulative_incidence" in curve.columns
    return checks


def run() -> int:
    started = time.time()
    run_id = os.getenv("RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    artifact_dir = Path(os.getenv("ARTIFACT_DIR", str(ROOT_DIR / "ARTIFACTS" / "validation")))
    scenario_artifacts = artifact_dir / "meta_stress"
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
            validate_no_derived_predictors(scenario)
            datasets, manifest = prepare_datasets(tmp_root / name, scenario)
            _assert_d2t_summary(manifest)

            model_checks = [
                run_model(name=name, datasets=datasets, scenario=scenario, model=model)
                for model in scenario["models"]
            ]
            checks = {
                "node_count": scenario["node_count"],
                "patients_per_node": scenario["patients_per_node"],
                "imputation_strategy": scenario["imputation_strategy"],
                "expected": scenario["expected"],
                "data_profile": scenario["data_profile"],
                "d2t_criteria": manifest.get("overall", {}).get("d2t_criteria", {}),
                "models": model_checks,
            }
            artifact = {
                "run_id": run_id,
                "timestamp_utc": now_utc(),
                "scenario": name,
                "status": "pass",
                "duration_s": round(time.time() - s_start, 3),
                "checks": _json_safe(checks),
                "manifest": manifest,
            }
            (scenario_artifacts / f"{name}.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
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
            (scenario_artifacts / f"{name}.json").write_text(json.dumps(fail_artifact, indent=2), encoding="utf-8")
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
                "artifact_dir": str(scenario_artifacts),
                "scenarios": [_json_safe(result.__dict__) for result in results],
                "error_summary": "; ".join(lane_errors),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0 if lane_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(run())
