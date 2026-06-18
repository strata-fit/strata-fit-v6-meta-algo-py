#!/usr/bin/env python3
"""Submit meta-algo smoke tasks against local vantage6 infra and compare to mock baseline."""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from vantage6.client import Client

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strata_fit_v6_meta_algo_py import run_local_meta_algorithm
from tests.infra.meta_stress_scenarios import (
    get_scenario,
    validate_no_derived_predictors,
)

TERMINAL_STATUSES = {
    "completed",
    "crashed",
    "failed",
    "cancelled",
    "non-existing Docker image",
}

IMPUTATION_COLUMNS = ["DAS28", "CRP", "HAQ", "Pat_global", "Ph_global", "Pain", "eq5d"]
COX_EXPL_VARS = ["Age_diagnosis", "DAS28", "CRP", "HAQ", "RF_positivity"]


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def env_csv(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)

    parsed = [value.strip() for value in raw.split(",") if value.strip()]
    return parsed or list(default)


def env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw is not None else default


def _load_scenario(name: str) -> dict[str, Any] | None:
    if not name or name == "infra_smoke":
        return None
    scenario = get_scenario(name)
    validate_no_derived_predictors(scenario)
    return scenario


def decode_result(value: Any) -> Any:
    if value is None or isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {"raw": str(value)}

    try:
        return json.loads(base64.b64decode(value).decode("utf-8"))
    except Exception:
        pass

    try:
        return json.loads(value)
    except Exception:
        return {"raw": value}


def wait_for_terminal(client: Client, task_id: int, timeout_s: int) -> str:
    status = None
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        current = client.task.get(task_id).get("status")
        if current != status:
            print(f"task {task_id} status: {current}")
            status = current
        if current in TERMINAL_STATUSES:
            return current
        time.sleep(2)
    raise TimeoutError(f"Task {task_id} did not finish before timeout")


def get_child_tasks(client: Client, parent_task_id: int) -> list[dict[str, Any]]:
    page = 1
    all_children: list[dict[str, Any]] = []
    while True:
        payload = client.task.list(parent=parent_task_id, page=page, per_page=50)
        chunk = payload.get("data", [])
        all_children.extend(chunk)
        if len(chunk) < 50:
            break
        page += 1
    return all_children


def assert_all_child_runs_completed(client: Client, parent_task_id: int) -> None:
    children = get_child_tasks(client, parent_task_id)
    if not children:
        raise RuntimeError(f"Task {parent_task_id} has no child tasks")

    statuses: list[str] = []
    run_count = 0
    for child in children:
        child_runs = client.run.from_task(child["id"]).get("data", [])
        run_count += len(child_runs)
        statuses.extend([run.get("status") for run in child_runs])

    if run_count == 0:
        raise RuntimeError(f"Task {parent_task_id} has no child runs")

    failed = [status for status in statuses if status != "completed"]
    if failed:
        raise RuntimeError(f"Task {parent_task_id} has non-completed child runs: {failed}")


def fetch_decoded_result(client: Client, task_id: int) -> dict[str, Any]:
    rows = client.result.from_task(task_id).get("data", [])
    if not rows:
        raise RuntimeError(f"Task {task_id} has no results")

    decoded = decode_result(rows[0].get("result"))
    if isinstance(decoded, dict) and decoded.get("ok") is False:
        raise RuntimeError(f"Task {task_id} returned error envelope: {decoded}")
    if not isinstance(decoded, dict):
        raise RuntimeError(f"Task {task_id} result is not a dict: {decoded}")
    return decoded


def authenticate_node_user(client: Client, node_name: str) -> None:
    client.authenticate(f"{node_name}-user", f"{node_name}-password")
    client.setup_encryption(None)


def authenticate_admin(client: Client) -> None:
    client.authenticate(
        env_str("V6_SERVER_USERNAME", "dev_admin"),
        env_str("V6_SERVER_PASSWORD", "password"),
    )
    client.setup_encryption(None)


def get_collaboration_org_map(client: Client, collaboration_id: int) -> dict[str, int]:
    rows = client.organization.list(collaboration=collaboration_id).get("data", [])
    return {row["name"]: row["id"] for row in rows if row.get("name") and row.get("id") is not None}


def create_task_with_master_fallback(
    *,
    client: Client,
    collab_id: int,
    master_candidates: list[str],
    org_map: dict[str, int],
    name: str,
    image: str,
    description: str,
    input_: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    errors: list[str] = []
    authenticate_admin(client)
    for master in master_candidates:
        payload = client.task.create(
            collaboration=collab_id,
            organizations=[org_map[master]],
            name=name,
            image=image,
            description=description,
            input_=input_,
            databases=[{"label": "default"}],
        )
        if isinstance(payload, dict) and payload.get("id") is not None:
            return payload, master
        errors.append(f"{master}: {payload}")

    joined = "; ".join(errors) if errors else "no task responses captured"
    raise RuntimeError(f"Failed to create task '{name}' with any candidate master ({joined})")


def _build_linear_input(
    org_ids: list[int],
    *,
    imputation_columns: list[str],
    imputation_strategy: str,
    final_model_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "master": True,
        "method": "main",
        "kwargs": {
            "columns": imputation_columns,
            "run_validation": True,
            "model_name": "PatientData",
            "imputation_strategy": imputation_strategy,
            "final_model": "sklearn_linear",
            "organizations": org_ids,
            "final_model_config": final_model_config,
        },
    }


def _build_cox_input(
    org_ids: list[int],
    *,
    imputation_columns: list[str],
    imputation_strategy: str,
    final_model_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "master": True,
        "method": "main",
        "kwargs": {
            "columns": imputation_columns,
            "run_validation": True,
            "model_name": "PatientData",
            "imputation_strategy": imputation_strategy,
            "final_model": "cox",
            "organizations": org_ids,
            "final_model_config": final_model_config,
        },
    }


def _build_km_input(
    org_ids: list[int],
    *,
    imputation_columns: list[str],
    imputation_strategy: str,
    final_model_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "master": True,
        "method": "main",
        "kwargs": {
            "columns": imputation_columns,
            "run_validation": True,
            "model_name": "PatientData",
            "imputation_strategy": imputation_strategy,
            "final_model": "km",
            "organizations": org_ids,
            "final_model_config": final_model_config,
        },
    }


def _build_mock_reference(
    data_paths: list[Path],
    run_linear: bool,
    run_cox: bool,
    run_km: bool,
    *,
    imputation_columns: list[str],
    imputation_strategy: str,
    model_configs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    datasets = [pd.read_csv(path) for path in data_paths]

    out: dict[str, dict[str, Any]] = {}
    if run_linear:
        out["sklearn_linear"] = run_local_meta_algorithm(
            datasets,
            columns=imputation_columns,
            run_validation=True,
            imputation_strategy=imputation_strategy,
            final_model="sklearn_linear",
            final_model_config=model_configs["sklearn_linear"],
        )
    if run_cox:
        out["cox"] = run_local_meta_algorithm(
            datasets,
            columns=imputation_columns,
            run_validation=True,
            imputation_strategy=imputation_strategy,
            final_model="cox",
            final_model_config=model_configs["cox"],
        )
    if run_km:
        out["km"] = run_local_meta_algorithm(
            datasets,
            columns=imputation_columns,
            run_validation=True,
            imputation_strategy=imputation_strategy,
            final_model="km",
            final_model_config=model_configs["km"],
        )
    return out


def _write_artifact(path: str, payload: dict[str, Any]) -> None:
    artifact_path = Path(path).expanduser()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _assert_cox_similarity(mock_result: dict[str, Any], infra_result: dict[str, Any]) -> None:
    mock_table = pd.read_json(StringIO(mock_result["final_result"]["model"]))
    infra_table = pd.read_json(StringIO(infra_result["final_result"]["model"]))
    mock_table = mock_table.sort_index()
    infra_table = infra_table.sort_index()

    if list(mock_table.index) != list(infra_table.index):
        raise RuntimeError("Cox covariates differ between mock and infra outputs")

    mock_coef = mock_table["Coef"].to_numpy(dtype=float)
    infra_coef = infra_table["Coef"].to_numpy(dtype=float)
    if not np.allclose(mock_coef, infra_coef, rtol=1e-4, atol=1e-4):
        raise RuntimeError(f"Cox coefficient mismatch mock={mock_coef} infra={infra_coef}")


def _assert_km_similarity(mock_result: dict[str, Any], infra_result: dict[str, Any]) -> None:
    mock_curve = pd.read_json(StringIO(mock_result["final_result"]["km_curve"]))
    infra_curve = pd.read_json(StringIO(infra_result["final_result"]["km_curve"]))

    if mock_curve.empty or infra_curve.empty:
        raise RuntimeError("KM curve empty in mock or infra result")

    mock_curve = mock_curve.sort_values("interval_start").reset_index(drop=True)
    infra_curve = infra_curve.sort_values("interval_start").reset_index(drop=True)

    if len(mock_curve) != len(infra_curve):
        raise RuntimeError(f"KM row count mismatch mock={len(mock_curve)} infra={len(infra_curve)}")

    mock_ci = mock_curve["cumulative_incidence"].to_numpy(dtype=float)
    infra_ci = infra_curve["cumulative_incidence"].to_numpy(dtype=float)
    if not np.allclose(mock_ci, infra_ci, rtol=1e-4, atol=1e-4):
        raise RuntimeError("KM cumulative incidence mismatch between mock and infra")


def main() -> None:
    host = os.getenv("V6_SERVER_HOST", "http://localhost")
    port = env_int("V6_SERVER_PORT", 5070)
    api_path = os.getenv("V6_API_PATH", "/api")
    collaboration_name = os.getenv("V6_COLLABORATION_NAME", "meta-ci")
    image = os.environ["V6_ALGO_IMAGE"]

    node_count = env_int("V6_NODE_COUNT", 3)
    timeout_s = env_int("V6_TASK_TIMEOUT_S", 1200)
    run_linear = env_bool("V6_RUN_LINEAR", True)
    run_cox = env_bool("V6_RUN_COX", True)
    run_km = env_bool("V6_RUN_KM", True)
    scenario_name = os.getenv("V6_SCENARIO_NAME", "infra_smoke")
    scenario_config = _load_scenario(scenario_name)
    result_artifact = os.getenv("V6_RESULT_ARTIFACT", "").strip()
    data_manifest_path = os.getenv("V6_DATA_MANIFEST_PATH", "").strip()
    imputation_strategy = os.getenv("V6_IMPUTATION_STRATEGY", "mean").strip() or "mean"
    imputation_columns = env_csv("V6_IMPUTATION_COLUMNS", IMPUTATION_COLUMNS)
    cox_expl_vars = env_csv("V6_COX_EXPL_VARS", COX_EXPL_VARS)
    model_configs: dict[str, dict[str, Any]] = {
        "sklearn_linear": {
            "predictors": ["Age_diagnosis", "DAS28", "CRP", "HAQ"],
            "outcome": "RF_positivity",
            "n_local_iterations": 25,
            "model_kwargs": {"solver": "lbfgs"},
        },
        "cox": {
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": cox_expl_vars,
            "max_iterations": 12,
            "tolerance": 1e-6,
            "preprocess_raw_data": True,
        },
        "km": {"preprocess_raw_data": True},
    }
    if scenario_config:
        models = set(scenario_config["models"])
        run_linear = "sklearn_linear" in models
        run_cox = "cox" in models
        run_km = "km" in models
        imputation_strategy = scenario_config["imputation_strategy"]
        imputation_columns = scenario_config["imputation_columns"]
        model_configs = scenario_config["model_configs"]

    ordered_names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    selected = ordered_names[:node_count]
    if len(selected) < 1:
        raise ValueError("V6_NODE_COUNT must be >= 1")

    data_dir_raw = os.getenv("V6_DATA_DIR")
    if not data_dir_raw:
        raise ValueError("V6_DATA_DIR must be provided for mock-vs-infra comparison")
    data_dir = Path(data_dir_raw).expanduser()
    data_paths = [data_dir / f"data_bucket{i + 1}.csv" for i in range(node_count)]
    for path in data_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing data partition: {path}")

    if not run_linear and not run_cox and not run_km:
        raise ValueError("Enable at least one of V6_RUN_LINEAR, V6_RUN_COX, or V6_RUN_KM")

    print("building mock baseline on the same partitions")
    mock_results = _build_mock_reference(
        data_paths,
        run_linear,
        run_cox,
        run_km,
        imputation_columns=imputation_columns,
        imputation_strategy=imputation_strategy,
        model_configs=model_configs,
    )

    client = Client(host, port, api_path)
    master_candidates = sorted(selected, key=lambda name: (name != "gamma", name))

    collab = None
    org_map: dict[str, int] = {}
    bootstrap_errors: list[str] = []
    try:
        authenticate_admin(client)
        collab = next(
            c for c in client.collaboration.list()["data"] if c["name"] == collaboration_name
        )
        org_map = get_collaboration_org_map(client, collab["id"])
    except Exception as exc:  # pragma: no cover
        bootstrap_errors.append(f"admin: {exc}")

    if collab is None:
        joined = "; ".join(bootstrap_errors) if bootstrap_errors else "no bootstrap attempts made"
        raise RuntimeError(f"Unable to bootstrap client session ({joined})")

    missing_orgs = [name for name in selected if name not in org_map]
    if missing_orgs:
        available = ", ".join(sorted(org_map)) or "<none>"
        raise RuntimeError(
            f"Collaboration '{collaboration_name}' is missing expected organizations {missing_orgs}. "
            f"Available collaboration organizations: {available}"
        )

    collab_id = collab["id"]
    org_ids = [org_map[name] for name in selected]
    summary: dict[str, Any] = {
        "scenario": scenario_name,
        "status": "pass",
        "image": image,
        "node_count": node_count,
        "selected_nodes": selected,
        "organization_ids": org_ids,
        "run_linear": run_linear,
        "run_cox": run_cox,
        "run_km": run_km,
        "imputation_strategy": imputation_strategy,
        "imputation_columns": imputation_columns,
        "model_configs": model_configs,
        "data_manifest_path": data_manifest_path,
        "tasks": {},
    }

    if run_linear:
        task, master = create_task_with_master_fallback(
            client=client,
            collab_id=collab_id,
            master_candidates=master_candidates,
            org_map=org_map,
            name=f"meta-linear-{node_count}nodes",
            image=image,
            description="meta sklearn_linear smoke",
            input_=_build_linear_input(
                org_ids,
                imputation_columns=imputation_columns,
                imputation_strategy=imputation_strategy,
                final_model_config=model_configs["sklearn_linear"],
            ),
        )
        task_id = task["id"]
        print(f"created linear task {task_id} orgs={selected} master={master}")
        status = wait_for_terminal(client, task_id, timeout_s)
        if status != "completed":
            raise RuntimeError(f"Linear task {task_id} finished with status '{status}'")

        decoded = fetch_decoded_result(client, task_id)
        if decoded.get("final_model") != "sklearn_linear":
            raise RuntimeError(f"Linear task {task_id} wrong final_model: {decoded.get('final_model')}")

        assert_all_child_runs_completed(client, task_id)
        summary["tasks"]["sklearn_linear"] = {
            "task_id": task_id,
            "status": status,
            "final_model": decoded.get("final_model"),
            "child_runs_completed": True,
        }
        print(f"linear task {task_id} validated")

    if run_cox:
        task, master = create_task_with_master_fallback(
            client=client,
            collab_id=collab_id,
            master_candidates=master_candidates,
            org_map=org_map,
            name=f"meta-cox-{node_count}nodes",
            image=image,
            description="meta cox smoke",
            input_=_build_cox_input(
                org_ids,
                imputation_columns=imputation_columns,
                imputation_strategy=imputation_strategy,
                final_model_config=model_configs["cox"],
            ),
        )
        task_id = task["id"]
        print(f"created cox task {task_id} orgs={selected} master={master}")
        status = wait_for_terminal(client, task_id, timeout_s)
        if status != "completed":
            raise RuntimeError(f"Cox task {task_id} finished with status '{status}'")

        decoded = fetch_decoded_result(client, task_id)
        if decoded.get("final_model") != "cox":
            raise RuntimeError(f"Cox task {task_id} wrong final_model: {decoded.get('final_model')}")

        _assert_cox_similarity(mock_results["cox"], decoded)
        assert_all_child_runs_completed(client, task_id)
        summary["tasks"]["cox"] = {
            "task_id": task_id,
            "status": status,
            "final_model": decoded.get("final_model"),
            "child_runs_completed": True,
            "included_organizations": decoded.get("final_result", {}).get("included_organizations", []),
        }
        print(f"cox task {task_id} validated and matched mock baseline")

    if run_km:
        task, master = create_task_with_master_fallback(
            client=client,
            collab_id=collab_id,
            master_candidates=master_candidates,
            org_map=org_map,
            name=f"meta-km-{node_count}nodes",
            image=image,
            description="meta km smoke",
            input_=_build_km_input(
                org_ids,
                imputation_columns=imputation_columns,
                imputation_strategy=imputation_strategy,
                final_model_config=model_configs["km"],
            ),
        )
        task_id = task["id"]
        print(f"created km task {task_id} orgs={selected} master={master}")
        status = wait_for_terminal(client, task_id, timeout_s)
        if status != "completed":
            raise RuntimeError(f"KM task {task_id} finished with status '{status}'")

        decoded = fetch_decoded_result(client, task_id)
        if decoded.get("final_model") != "km":
            raise RuntimeError(f"KM task {task_id} wrong final_model: {decoded.get('final_model')}")

        _assert_km_similarity(mock_results["km"], decoded)
        assert_all_child_runs_completed(client, task_id)
        summary["tasks"]["km"] = {
            "task_id": task_id,
            "status": status,
            "final_model": decoded.get("final_model"),
            "child_runs_completed": True,
            "included_organizations": decoded.get("final_result", {}).get("included_organizations", []),
        }
        print(f"km task {task_id} validated and matched mock baseline")

    if result_artifact:
        _write_artifact(result_artifact, summary)
    print("meta-algo infra smoke tasks completed")


if __name__ == "__main__":
    main()
