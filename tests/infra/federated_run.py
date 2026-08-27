#!/usr/bin/env python3
"""Run one-node federated KM and Cox flows with MICE and persist run manifests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from vantage6.client import Client

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.infra.run_algo_smoke import (
    _build_cox_input,
    _build_km_input,
    assert_all_child_runs_completed,
    fetch_decoded_result,
    wait_for_terminal,
)

DEFAULT_IMPUTATION_COLUMNS = ["DAS28", "CRP", "HAQ", "Pat_global", "Ph_global", "Pain", "eq5d"]
DEFAULT_COX_EXPL_VARS = ["Age_diagnosis", "DAS28", "CRP", "HAQ", "RF_positivity"]


# ============================================================================
# Connection and credentials
# ============================================================================
V6_SERVER_HOST = os.getenv("V6_SERVER_HOST", "http://localhost")
V6_SERVER_PORT = int(os.getenv("V6_SERVER_PORT", "5070"))
V6_API_PATH = os.getenv("V6_API_PATH", "/api")

V6_USERNAME = os.getenv("V6_USERNAME", "alpha-user")
V6_PASSWORD = os.getenv("V6_PASSWORD", "alpha-password")
V6_MFA_CODE = os.getenv("V6_MFA_CODE", "")
V6_USER_PRIVATE_KEY = os.getenv("V6_USER_PRIVATE_KEY", "")


# ============================================================================
# Collaboration, organizations, image
# ============================================================================
V6_COLLABORATION_ID = int(os.getenv("V6_COLLABORATION_ID", "1"))
V6_MASTER_ORG_ID = int(os.getenv("V6_MASTER_ORG_ID", "1"))
V6_ORGANIZATION_IDS = [
    int(value.strip())
    for value in os.getenv("V6_ORGANIZATION_IDS", str(V6_MASTER_ORG_ID)).split(",")
    if value.strip()
]

V6_ALGO_IMAGE = os.getenv(
    "V6_ALGO_IMAGE", "localhost:5000/strata-fit-v6-meta-algo:single-node-mice-v9"
)
V6_DATABASE_LABEL = os.getenv("V6_DATABASE_LABEL", "default")
V6_TASK_TIMEOUT_S = int(os.getenv("V6_TASK_TIMEOUT_S", "1200"))


# ============================================================================
# Data generation and manifests
# ============================================================================
RUN_GENERATE_DATA = os.getenv("RUN_GENERATE_DATA", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/meta_single_node_mice/data")).expanduser()
MANIFEST_DIR = Path(
    os.getenv("MANIFEST_DIR", "/tmp/meta_single_node_mice/manifests")
).expanduser()

DATA_NODE_COUNT = int(os.getenv("DATA_NODE_COUNT", "1"))
DATA_PATIENTS_PER_NODE = int(os.getenv("DATA_PATIENTS_PER_NODE", "24"))
DATA_ROWS_PER_NODE = int(os.getenv("DATA_ROWS_PER_NODE", "100"))
DATA_SEED = int(os.getenv("DATA_SEED", "20260318"))
DATA_MAX_ATTEMPTS = int(os.getenv("DATA_MAX_ATTEMPTS", "20"))
DATA_MIN_C_INDEX = float(os.getenv("DATA_MIN_C_INDEX", "0.60"))
DATA_MAX_LOGRANK_P = float(os.getenv("DATA_MAX_LOGRANK_P", "0.10"))
PYTHON_BIN = os.getenv("PYTHON_BIN", sys.executable)


# ============================================================================
# Model/runtime parameters
# ============================================================================
IMPUTATION_STRATEGY = os.getenv("IMPUTATION_STRATEGY", "mice").strip() or "mice"
IMPUTATION_COLUMNS = [
    value.strip()
    for value in os.getenv("IMPUTATION_COLUMNS", ",".join(DEFAULT_IMPUTATION_COLUMNS)).split(",")
    if value.strip()
] or list(DEFAULT_IMPUTATION_COLUMNS)
COX_EXPL_VARS = [
    value.strip()
    for value in os.getenv("COX_EXPL_VARS", ",".join(DEFAULT_COX_EXPL_VARS)).split(",")
    if value.strip()
] or list(DEFAULT_COX_EXPL_VARS)

KM_TASK_NAME = os.getenv("KM_TASK_NAME", "meta-km-1node-mice-federated-run")
COX_TASK_NAME = os.getenv("COX_TASK_NAME", "meta-cox-1node-mice-federated-run")


def _is_placeholder(value: str) -> bool:
    value = value.strip()
    return value.startswith("<") and value.endswith(">")


def _assert_required_constants() -> None:
    required = {
        "V6_SERVER_HOST": V6_SERVER_HOST,
        "V6_USERNAME": V6_USERNAME,
        "V6_PASSWORD": V6_PASSWORD,
        "V6_ALGO_IMAGE": V6_ALGO_IMAGE,
        "V6_DATABASE_LABEL": V6_DATABASE_LABEL,
    }
    missing = [
        key
        for key, value in required.items()
        if not str(value).strip() or _is_placeholder(str(value))
    ]
    if missing:
        raise ValueError(
            "Missing required configuration values: "
            + ", ".join(missing)
            + ". Fill constants at the top of this script or export env vars."
        )

    if not V6_ORGANIZATION_IDS:
        raise ValueError("V6_ORGANIZATION_IDS must contain at least one organization id")


def _authenticate(client: Client) -> None:
    if V6_MFA_CODE.strip():
        client.authenticate(V6_USERNAME, V6_PASSWORD, mfa_code=V6_MFA_CODE)
    else:
        client.authenticate(V6_USERNAME, V6_PASSWORD)

    if V6_USER_PRIVATE_KEY.strip():
        client.setup_encryption(V6_USER_PRIVATE_KEY)
    else:
        client.setup_encryption(None)


def _run_data_generation(data_manifest_path: Path) -> None:
    cmd = [
        PYTHON_BIN,
        str(ROOT / "tests/infra/prepare_meta_smoke_data.py"),
        "--output-dir",
        str(DATA_DIR),
        "--node-count",
        str(DATA_NODE_COUNT),
        "--patients-per-node",
        str(DATA_PATIENTS_PER_NODE),
        "--rows-per-node",
        str(DATA_ROWS_PER_NODE),
        "--seed",
        str(DATA_SEED),
        "--max-attempts",
        str(DATA_MAX_ATTEMPTS),
        "--min-c-index",
        str(DATA_MIN_C_INDEX),
        "--max-logrank-p",
        str(DATA_MAX_LOGRANK_P),
        "--manifest-path",
        str(data_manifest_path),
    ]
    subprocess.run(cmd, check=True)


def _create_task(
    *,
    client: Client,
    task_name: str,
    description: str,
    input_: dict[str, Any],
) -> dict[str, Any]:
    payload = client.task.create(
        collaboration=V6_COLLABORATION_ID,
        organizations=[V6_MASTER_ORG_ID],
        name=task_name,
        image=V6_ALGO_IMAGE,
        description=description,
        input_=input_,
        databases=[{"label": V6_DATABASE_LABEL}],
    )
    if not isinstance(payload, dict) or payload.get("id") is None:
        raise RuntimeError(f"Task creation failed for '{task_name}': {payload}")
    return payload


def _assert_collaboration_membership(client: Client) -> None:
    org_rows = client.organization.list(collaboration=V6_COLLABORATION_ID).get("data", [])
    collab_org_ids = {int(row["id"]) for row in org_rows if row.get("id") is not None}

    if V6_MASTER_ORG_ID not in collab_org_ids:
        raise ValueError(
            f"V6_MASTER_ORG_ID={V6_MASTER_ORG_ID} is not in collaboration id={V6_COLLABORATION_ID} "
            f"organization ids {sorted(collab_org_ids)}"
        )

    missing = [org_id for org_id in V6_ORGANIZATION_IDS if org_id not in collab_org_ids]
    if missing:
        raise ValueError(
            f"V6_ORGANIZATION_IDS contains ids not in collaboration id={V6_COLLABORATION_ID}: {missing}"
        )


def main() -> None:
    _assert_required_constants()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    data_manifest_path = MANIFEST_DIR / "run_manifest.json"
    federated_result_path = MANIFEST_DIR / "federated_results.json"

    if RUN_GENERATE_DATA:
        _run_data_generation(data_manifest_path)

    print("Connecting to Vantage6 server...")
    client = Client(V6_SERVER_HOST, V6_SERVER_PORT, V6_API_PATH)
    _authenticate(client)
    _assert_collaboration_membership(client)

    summary: dict[str, Any] = {
        "image": V6_ALGO_IMAGE,
        "collaboration_id": V6_COLLABORATION_ID,
        "master_org_id": V6_MASTER_ORG_ID,
        "organization_ids": V6_ORGANIZATION_IDS,
        "data_dir": str(DATA_DIR),
        "data_manifest_path": str(data_manifest_path),
    }

    km_task = _create_task(
        client=client,
        task_name=KM_TASK_NAME,
        description="KM federated run",
        input_=_build_km_input(
            V6_ORGANIZATION_IDS,
            imputation_columns=IMPUTATION_COLUMNS,
            imputation_strategy=IMPUTATION_STRATEGY,
        ),
    )
    km_task_id = int(km_task["id"])
    km_status = wait_for_terminal(client, km_task_id, V6_TASK_TIMEOUT_S)
    if km_status != "completed":
        raise RuntimeError(f"KM task {km_task_id} finished with status '{km_status}'")

    km_decoded = fetch_decoded_result(client, km_task_id)
    assert_all_child_runs_completed(client, km_task_id)
    km_curve = pd.read_json(StringIO(km_decoded["final_result"]["km_curve"]))
    km_curve = km_curve.sort_values("interval_start").reset_index(drop=True)

    summary["km"] = {
        "task_id": km_task_id,
        "status": km_status,
        "final_model": km_decoded.get("final_model"),
        "rows": int(len(km_curve)),
        "final_cumulative_incidence": float(km_curve["cumulative_incidence"].iloc[-1]),
        "included_organizations": km_decoded["final_result"].get("included_organizations", []),
    }

    cox_task = _create_task(
        client=client,
        task_name=COX_TASK_NAME,
        description="Cox federated run",
        input_=_build_cox_input(
            V6_ORGANIZATION_IDS,
            imputation_columns=IMPUTATION_COLUMNS,
            imputation_strategy=IMPUTATION_STRATEGY,
            cox_expl_vars=COX_EXPL_VARS,
        ),
    )
    cox_task_id = int(cox_task["id"])
    cox_status = wait_for_terminal(client, cox_task_id, V6_TASK_TIMEOUT_S)
    if cox_status != "completed":
        raise RuntimeError(f"Cox task {cox_task_id} finished with status '{cox_status}'")

    cox_decoded = fetch_decoded_result(client, cox_task_id)
    assert_all_child_runs_completed(client, cox_task_id)
    cox_model = pd.read_json(StringIO(cox_decoded["final_result"]["model"])).sort_index()
    p_col = "P-value" if "P-value" in cox_model.columns else "p-value"

    summary["cox"] = {
        "task_id": cox_task_id,
        "status": cox_status,
        "final_model": cox_decoded.get("final_model"),
        "included_organizations": cox_decoded["final_result"].get("included_organizations", []),
        "overall_p_value": float(cox_decoded["final_result"].get("overall_p_value")),
        "aic": float(cox_decoded["final_result"].get("aic")),
        "coef": {k: float(v) for k, v in cox_model["Coef"].to_dict().items()},
        "p_values": {k: float(v) for k, v in cox_model[p_col].to_dict().items()},
    }

    with federated_result_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"wrote {federated_result_path}")


if __name__ == "__main__":
    main()
