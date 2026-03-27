#!/usr/bin/env python3
"""Run one-node federated KM and Cox flows with MICE and persist run manifests."""

from __future__ import annotations

import argparse
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
    authenticate_node_user,
    create_task_with_master_fallback,
    fetch_decoded_result,
    wait_for_terminal,
)

DEFAULT_IMPUTATION_COLUMNS = ["DAS28", "CRP", "HAQ", "Pat_global", "Ph_global", "Pain", "eq5d"]
DEFAULT_COX_EXPL_VARS = ["Age_diagnosis", "DAS28", "CRP", "HAQ", "RF_positivity"]


def parse_csv(raw: str, fallback: list[str]) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values or list(fallback)


def run_data_generation(
    *,
    python_bin: str,
    data_dir: Path,
    manifest_path: Path,
    node_count: int,
    patients_per_node: int,
    rows_per_node: int,
    seed: int,
    max_attempts: int,
    min_c_index: float,
    max_logrank_p: float,
) -> None:
    cmd = [
        python_bin,
        str(ROOT / "tests/infra/prepare_meta_smoke_data.py"),
        "--output-dir",
        str(data_dir),
        "--node-count",
        str(node_count),
        "--patients-per-node",
        str(patients_per_node),
        "--rows-per-node",
        str(rows_per_node),
        "--seed",
        str(seed),
        "--max-attempts",
        str(max_attempts),
        "--min-c-index",
        str(min_c_index),
        "--max-logrank-p",
        str(max_logrank_p),
        "--manifest-path",
        str(manifest_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--host", default="http://localhost")
    parser.add_argument("--port", type=int, default=5070)
    parser.add_argument("--api-path", default="/api")
    parser.add_argument("--collaboration-name", default="meta-single-node-mice")
    parser.add_argument("--image", required=True)
    parser.add_argument("--node-name", default="alpha")
    parser.add_argument("--timeout-s", type=int, default=1200)

    parser.add_argument("--data-dir", default="/tmp/meta_single_node_mice_20260326/data")
    parser.add_argument("--manifest-dir", default="/tmp/meta_single_node_mice_20260326/manifests")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--node-count", type=int, default=1)
    parser.add_argument("--patients-per-node", type=int, default=24)
    parser.add_argument("--rows-per-node", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260318)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--min-c-index", type=float, default=0.60)
    parser.add_argument("--max-logrank-p", type=float, default=0.10)

    parser.add_argument(
        "--imputation-columns",
        default=",".join(DEFAULT_IMPUTATION_COLUMNS),
        help="Comma-separated columns for MICE metrics/imputation.",
    )
    parser.add_argument(
        "--cox-expl-vars",
        default=",".join(DEFAULT_COX_EXPL_VARS),
        help="Comma-separated Cox explanatory variables.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    manifest_dir = Path(args.manifest_dir).expanduser()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    data_manifest_path = manifest_dir / "run_manifest.json"
    federated_result_path = manifest_dir / "federated_results.json"

    if not args.skip_generate:
        run_data_generation(
            python_bin=args.python_bin,
            data_dir=data_dir,
            manifest_path=data_manifest_path,
            node_count=args.node_count,
            patients_per_node=args.patients_per_node,
            rows_per_node=args.rows_per_node,
            seed=args.seed,
            max_attempts=args.max_attempts,
            min_c_index=args.min_c_index,
            max_logrank_p=args.max_logrank_p,
        )

    imputation_columns = parse_csv(args.imputation_columns, DEFAULT_IMPUTATION_COLUMNS)
    cox_expl_vars = parse_csv(args.cox_expl_vars, DEFAULT_COX_EXPL_VARS)

    client = Client(args.host, args.port, args.api_path)
    authenticate_node_user(client, args.node_name)
    collab = next(
        c for c in client.collaboration.list()["data"] if c["name"] == args.collaboration_name
    )
    org_map = {o["name"]: o["id"] for o in client.organization.list()["data"]}
    org_ids = [org_map[args.node_name]]

    summary: dict[str, Any] = {
        "image": args.image,
        "node_name": args.node_name,
        "collaboration_name": args.collaboration_name,
        "data_dir": str(data_dir),
        "data_manifest_path": str(data_manifest_path),
    }

    km_task, km_master = create_task_with_master_fallback(
        client=client,
        collab_id=collab["id"],
        master_candidates=[args.node_name],
        org_map=org_map,
        name="meta-km-1node-mice-federated-run",
        image=args.image,
        description="KM federated run",
        input_=_build_km_input(
            org_ids,
            imputation_columns=imputation_columns,
            imputation_strategy="mice",
        ),
    )
    km_task_id = km_task["id"]
    km_status = wait_for_terminal(client, km_task_id, args.timeout_s)
    km_decoded = fetch_decoded_result(client, km_task_id)
    assert_all_child_runs_completed(client, km_task_id)
    km_curve = pd.read_json(StringIO(km_decoded["final_result"]["km_curve"]))
    km_curve = km_curve.sort_values("interval_start").reset_index(drop=True)

    summary["km"] = {
        "task_id": km_task_id,
        "status": km_status,
        "master": km_master,
        "final_model": km_decoded.get("final_model"),
        "rows": int(len(km_curve)),
        "final_cumulative_incidence": float(km_curve["cumulative_incidence"].iloc[-1]),
        "included_organizations": km_decoded["final_result"].get("included_organizations", []),
    }

    cox_task, cox_master = create_task_with_master_fallback(
        client=client,
        collab_id=collab["id"],
        master_candidates=[args.node_name],
        org_map=org_map,
        name="meta-cox-1node-mice-federated-run",
        image=args.image,
        description="Cox federated run",
        input_=_build_cox_input(
            org_ids,
            imputation_columns=imputation_columns,
            imputation_strategy="mice",
            cox_expl_vars=cox_expl_vars,
        ),
    )
    cox_task_id = cox_task["id"]
    cox_status = wait_for_terminal(client, cox_task_id, args.timeout_s)
    cox_decoded = fetch_decoded_result(client, cox_task_id)
    assert_all_child_runs_completed(client, cox_task_id)
    cox_model = pd.read_json(StringIO(cox_decoded["final_result"]["model"])).sort_index()
    p_col = "P-value" if "P-value" in cox_model.columns else "p-value"

    summary["cox"] = {
        "task_id": cox_task_id,
        "status": cox_status,
        "master": cox_master,
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
