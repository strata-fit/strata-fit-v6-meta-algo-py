#!/usr/bin/env python3
"""Generate synthetic STRATA-FIT CSV buckets for meta-algo infra smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tests.synthetic_strata_fit_data import (
    SyntheticConfig,
    generate_d2t_truth_table_dataset,
    write_partitioned_csv,
)
from tests.infra.meta_stress_scenarios import get_scenario, validate_no_derived_predictors
from strata_fit_v6_meta_algo_py.preprocessing import (
    derive_d2t_ra_visit_flags,
    strata_fit_data_to_cox_input,
)

COX_EXPL_VARS = ["Age_diagnosis", "DAS28", "CRP", "HAQ", "RF_positivity"]
IMPUTATION_COLUMNS = ["DAS28", "CRP", "HAQ", "Pat_global", "Ph_global", "Pain", "eq5d"]


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truncate_to_rows(path: Path, rows_per_node: int) -> None:
    if rows_per_node <= 0:
        return
    df = pd.read_csv(path)
    if len(df) <= rows_per_node:
        return
    df.head(rows_per_node).to_csv(path, index=False)


def _harrell_c_index(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float | None:
    concordant = 0.0
    comparable = 0
    n = len(times)
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = times[i], times[j]
            ei, ej = events[i], events[j]
            ri, rj = risks[i], risks[j]

            if ei == 1 and ti < tj:
                comparable += 1
                if ri > rj:
                    concordant += 1.0
                elif ri == rj:
                    concordant += 0.5
            elif ej == 1 and tj < ti:
                comparable += 1
                if rj > ri:
                    concordant += 1.0
                elif ri == rj:
                    concordant += 0.5

    if comparable == 0:
        return None
    return concordant / comparable


def _logrank_high_low(
    times: np.ndarray,
    events: np.ndarray,
    high_group: np.ndarray,
) -> tuple[float | None, float | None]:
    unique_event_times = sorted(set(times[events == 1].tolist()))
    if not unique_event_times:
        return None, None

    observed_minus_expected = 0.0
    variance_total = 0.0

    for current_time in unique_event_times:
        at_risk = times >= current_time
        n_high = int(np.sum(at_risk & high_group))
        n_low = int(np.sum(at_risk & ~high_group))
        n_total = n_high + n_low
        if n_total <= 1:
            continue

        at_time_event = (times == current_time) & (events == 1)
        d_high = int(np.sum(at_time_event & high_group))
        d_low = int(np.sum(at_time_event & ~high_group))
        d_total = d_high + d_low
        if d_total == 0:
            continue

        expected_high = d_total * (n_high / n_total)
        variance = (
            n_high * n_low * d_total * (n_total - d_total)
        ) / (n_total * n_total * (n_total - 1))

        observed_minus_expected += d_high - expected_high
        variance_total += variance

    if variance_total <= 0:
        return None, None

    z_score = observed_minus_expected / math.sqrt(variance_total)
    p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
    return z_score, p_value


def _risk_score(cox_df: pd.DataFrame) -> np.ndarray:
    feature_frame = cox_df[COX_EXPL_VARS].copy()
    for column in COX_EXPL_VARS:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
        feature_frame[column] = feature_frame[column].fillna(feature_frame[column].median())

    standardized = feature_frame.copy()
    for column in COX_EXPL_VARS:
        series = standardized[column]
        std = float(series.std(ddof=0))
        if std <= 1e-12:
            standardized[column] = 0.0
        else:
            standardized[column] = (series - float(series.mean())) / std

    weights = {
        "Age_diagnosis": 0.35,
        "DAS28": 0.90,
        "CRP": 0.40,
        "HAQ": 0.80,
        "RF_positivity": 0.25,
    }

    score = np.zeros(len(standardized), dtype=float)
    for column in COX_EXPL_VARS:
        score += weights.get(column, 0.30) * standardized[column].to_numpy(dtype=float)
    return score


def _signal_metrics(df: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    try:
        cox_df = strata_fit_data_to_cox_input(
            df,
            time_col="time",
            outcome_col="event",
            expl_vars=COX_EXPL_VARS,
        )
    except Exception as exc:
        return {"error": f"cox-preprocess-failed: {exc}"}

    if cox_df.empty:
        return {"error": "cox-preprocess-empty"}

    clean = cox_df.dropna(subset=["time", "event"]).copy()
    for column in COX_EXPL_VARS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    times = clean["time"].to_numpy(dtype=float)
    events = clean["event"].to_numpy(dtype=int)
    risks = _risk_score(clean)

    c_index = _harrell_c_index(times, events, risks)
    median_risk = float(np.median(risks))
    high_group = risks >= median_risk
    logrank_z, logrank_p = _logrank_high_low(times, events, high_group)

    metrics["n_patients"] = int(len(clean))
    metrics["n_events"] = int(np.sum(events))
    metrics["event_rate"] = _finite_or_none(np.mean(events) if len(events) else None)
    metrics["c_index"] = _finite_or_none(c_index)
    metrics["logrank_z"] = _finite_or_none(logrank_z)
    metrics["logrank_p"] = _finite_or_none(logrank_p)
    metrics["risk_group_sizes"] = {
        "high": int(np.sum(high_group)),
        "low": int(np.sum(~high_group)),
    }
    return metrics


def _d2t_criteria_summary(df: pd.DataFrame) -> dict[str, Any]:
    try:
        visits = derive_d2t_ra_visit_flags(df)
    except Exception as exc:
        return {"error": f"d2t-criteria-failed: {exc}"}

    criteria = ["D2T_crit1", "D2T_crit2", "D2T_crit3", "D2T_RA"]
    if visits.empty:
        return {
            "visits": {column: 0 for column in criteria},
            "patients": {column: 0 for column in criteria},
            "all_three_required": True,
        }
    missing = [column for column in criteria if column not in visits.columns]
    if missing:
        return {"error": f"d2t-criteria-missing-columns: {', '.join(missing)}"}
    out: dict[str, Any] = {
        "visits": {column: int(visits[column].sum()) for column in criteria},
        "patients": {},
        "all_three_required": bool(
            (visits["D2T_RA"] == (visits["D2T_crit1"] & visits["D2T_crit2"] & visits["D2T_crit3"])).all()
        ),
    }
    grouped = visits.groupby("pat_ID")[criteria].max()
    out["patients"] = {column: int(grouped[column].sum()) for column in criteria}
    return out


def _node_manifest(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    missingness = {
        column: _finite_or_none(float(df[column].isna().mean())) if column in df.columns else None
        for column in IMPUTATION_COLUMNS
    }
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "rows": int(len(df)),
        "unique_patients": int(df["pat_ID"].nunique()) if "pat_ID" in df.columns else None,
        "columns": list(df.columns),
        "missingness": missingness,
        "signal": _signal_metrics(df),
        "d2t_criteria": _d2t_criteria_summary(df),
    }


def _quality_pass(overall_signal: dict[str, Any], min_c_index: float, max_logrank_p: float) -> bool:
    c_index = overall_signal.get("c_index")
    logrank_p = overall_signal.get("logrank_p")
    if c_index is None or logrank_p is None:
        return False
    return c_index >= min_c_index and logrank_p <= max_logrank_p


def _build_manifest(
    *,
    paths: list[Path],
    node_count: int,
    patients_per_node: int,
    seed: int,
    rows_per_node: int,
    min_c_index: float,
    max_logrank_p: float,
    attempt: int,
) -> dict[str, Any]:
    nodes = [_node_manifest(path) for path in paths]
    combined = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    overall_signal = _signal_metrics(combined)
    quality_passed = _quality_pass(overall_signal, min_c_index=min_c_index, max_logrank_p=max_logrank_p)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt,
        "config": {
            "node_count": node_count,
            "patients_per_node": patients_per_node,
            "seed": seed,
            "rows_per_node": rows_per_node,
        },
        "quality_targets": {
            "min_c_index": min_c_index,
            "max_logrank_p": max_logrank_p,
        },
        "quality_passed": quality_passed,
        "overall": {
            "rows": int(len(combined)),
            "unique_patients": int(combined["pat_ID"].nunique()) if "pat_ID" in combined.columns else None,
            "signal": overall_signal,
            "d2t_criteria": _d2t_criteria_summary(combined),
        },
        "nodes": nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--node-count", type=int, default=3)
    parser.add_argument("--patients-per-node", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20260318)
    parser.add_argument("--rows-per-node", type=int, default=0)
    parser.add_argument("--min-c-index", type=float, default=0.60)
    parser.add_argument("--max-logrank-p", type=float, default=0.10)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--scenario", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_config: dict[str, Any] | None = None
    data_profile: dict[str, Any] = {}
    if args.scenario:
        scenario_config = get_scenario(args.scenario)
        validate_no_derived_predictors(scenario_config)
        data_profile = scenario_config["data_profile"]
        args.node_count = int(scenario_config["node_count"])
        args.patients_per_node = int(scenario_config["patients_per_node"])
        args.seed = int(scenario_config["seed"])

    selected_paths: list[Path] = []
    selected_manifest: dict[str, Any] | None = None
    skip_quality_loop = False

    if scenario_config and scenario_config["name"] == "d2t_criteria_truth_table":
        skip_quality_loop = True
        path = output_dir / "data_bucket1.csv"
        generate_d2t_truth_table_dataset().to_csv(path, index=False)
        selected_paths = [path]
        selected_manifest = _build_manifest(
            paths=selected_paths,
            node_count=1,
            patients_per_node=8,
            seed=args.seed,
            rows_per_node=args.rows_per_node,
            min_c_index=args.min_c_index,
            max_logrank_p=args.max_logrank_p,
            attempt=1,
        )
        selected_manifest["scenario"] = {
            key: value
            for key, value in scenario_config.items()
            if key not in {"model_configs"}
        }

    for attempt in range(1, args.max_attempts + 1):
        if skip_quality_loop:
            break
        candidate_seed = args.seed + (attempt - 1)
        paths = write_partitioned_csv(
            output_dir=output_dir,
            config=SyntheticConfig(
                node_count=args.node_count,
                patients_per_node=args.patients_per_node,
                seed=candidate_seed,
                missingness_profile=data_profile.get("missingness_profile", "baseline"),
                event_profile=data_profile.get("event_profile", "adequate"),
                signal_profile=data_profile.get("signal_profile", "strong"),
                site_heterogeneity=bool(data_profile.get("site_heterogeneity", False)),
            ),
        )
        for path in paths:
            _truncate_to_rows(path, args.rows_per_node)

        manifest = _build_manifest(
            paths=paths,
            node_count=args.node_count,
            patients_per_node=args.patients_per_node,
            seed=candidate_seed,
            rows_per_node=args.rows_per_node,
            min_c_index=args.min_c_index,
            max_logrank_p=args.max_logrank_p,
            attempt=attempt,
        )
        if scenario_config:
            manifest["scenario"] = {
                key: value
                for key, value in scenario_config.items()
                if key not in {"model_configs"}
            }
        selected_paths = paths
        selected_manifest = manifest

        signal = manifest["overall"]["signal"]
        print(
            "attempt "
            f"{attempt}: seed={candidate_seed} "
            f"c_index={signal.get('c_index')} "
            f"logrank_p={signal.get('logrank_p')} "
            f"quality_passed={manifest['quality_passed']}"
        )
        if manifest["quality_passed"]:
            break

    if selected_manifest is None:
        raise RuntimeError("No manifest produced during data generation")

    manifest_path = (
        Path(args.manifest_path).expanduser()
        if args.manifest_path
        else output_dir / "run_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(selected_manifest, handle, indent=2)

    for path in selected_paths:
        print(f"wrote {path}")
    print(f"manifest {manifest_path}")


if __name__ == "__main__":
    main()
