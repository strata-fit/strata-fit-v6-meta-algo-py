#!/usr/bin/env python3
"""Submit meta-algo smoke tasks against local vantage6 infra and validate completion."""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

from vantage6.client import Client

TERMINAL_STATUSES = {
    "completed",
    "crashed",
    "failed",
    "cancelled",
    "non-existing Docker image",
}


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
    for master in master_candidates:
        authenticate_node_user(client, master)
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

    ordered_names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    selected = ordered_names[:node_count]
    if len(selected) < 2:
        raise ValueError("V6_NODE_COUNT must be >= 2")

    client = Client(host, port, api_path)
    master_candidates = sorted(selected, key=lambda name: (name != "gamma", name))

    collab = None
    org_map: dict[str, int] = {}
    bootstrap_errors: list[str] = []
    for candidate in master_candidates:
        try:
            authenticate_node_user(client, candidate)
            collab = next(
                c for c in client.collaboration.list()["data"] if c["name"] == collaboration_name
            )
            org_map = {o["name"]: o["id"] for o in client.organization.list()["data"]}
            break
        except Exception as exc:  # pragma: no cover - defensive path for infra auth
            bootstrap_errors.append(f"{candidate}: {exc}")

    if collab is None:
        joined = "; ".join(bootstrap_errors) if bootstrap_errors else "no bootstrap attempts made"
        raise RuntimeError(f"Unable to bootstrap client session ({joined})")

    collab_id = collab["id"]
    org_ids = [org_map[name] for name in selected]

    if not run_linear and not run_cox:
        raise ValueError("At least one of V6_RUN_LINEAR or V6_RUN_COX must be true")

    if run_linear:
        linear_input = {
            "master": True,
            "method": "main",
            "kwargs": {
                "columns": ["DAS28", "CRP", "HAQ"],
                "run_validation": False,
                "imputation_strategy": "mean",
                "final_model": "sklearn_linear",
                "organizations": org_ids,
                "final_model_config": {
                    "predictors": ["Age_diagnosis", "DAS28", "CRP", "HAQ"],
                    "outcome": "RF_positivity",
                    "n_local_iterations": 25,
                    "model_kwargs": {"solver": "lbfgs"},
                },
            },
        }

        task, master = create_task_with_master_fallback(
            client=client,
            collab_id=collab_id,
            master_candidates=master_candidates,
            org_map=org_map,
            name=f"meta-linear-{node_count}nodes",
            image=image,
            description="meta sklearn_linear smoke",
            input_=linear_input,
        )

        task_id = task["id"]
        print(f"created linear task {task_id} orgs={selected} master={master}")
        status = wait_for_terminal(client, task_id, timeout_s)
        if status != "completed":
            raise RuntimeError(f"Linear task {task_id} finished with status '{status}'")

        decoded = fetch_decoded_result(client, task_id)
        if decoded.get("final_model") != "sklearn_linear":
            raise RuntimeError(f"Linear task {task_id} wrong final_model: {decoded.get('final_model')}")

        final_result = decoded.get("final_result", {})
        attrs = final_result.get("model_attributes", {})
        if "coef_" not in attrs or "intercept_" not in attrs:
            raise RuntimeError(f"Linear task {task_id} missing model attributes: {attrs}")

        partials = final_result.get("partials", [])
        if len(partials) != len(org_ids):
            raise RuntimeError(
                f"Linear task {task_id} expected {len(org_ids)} partials, got {len(partials)}"
            )

        assert_all_child_runs_completed(client, task_id)
        print(f"linear task {task_id} validated")

    if run_cox:
        cox_input = {
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
                    "max_iterations": 10,
                    "tolerance": 1e-6,
                },
            },
        }

        task, master = create_task_with_master_fallback(
            client=client,
            collab_id=collab_id,
            master_candidates=master_candidates,
            org_map=org_map,
            name=f"meta-cox-{node_count}nodes",
            image=image,
            description="meta cox smoke",
            input_=cox_input,
        )

        task_id = task["id"]
        print(f"created cox task {task_id} orgs={selected} master={master}")
        status = wait_for_terminal(client, task_id, timeout_s)
        if status != "completed":
            raise RuntimeError(f"Cox task {task_id} finished with status '{status}'")

        decoded = fetch_decoded_result(client, task_id)
        if decoded.get("final_model") != "cox":
            raise RuntimeError(f"Cox task {task_id} wrong final_model: {decoded.get('final_model')}")

        final_result = decoded.get("final_result", {})
        required = {
            "aic",
            "degrees_of_freedom",
            "excluded_organizations",
            "included_organizations",
            "model",
            "overall_p_value",
            "warnings",
        }
        missing = required - set(final_result.keys())
        if missing:
            raise RuntimeError(f"Cox task {task_id} missing keys: {sorted(missing)}")

        included = final_result.get("included_organizations", [])
        if not included:
            raise RuntimeError(f"Cox task {task_id} included_organizations is empty")

        assert_all_child_runs_completed(client, task_id)
        print(f"cox task {task_id} validated")

    print("meta-algo infra smoke tasks completed")


if __name__ == "__main__":
    main()
