"""
Central orchestrator for the STRATA-FIT meta-algorithm.
Exposes a Vantage6 entrypoint and keeps a local helper for mock testing.
"""
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd
from io import StringIO
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient

from strata_fit_v6_imputation_py.imputation_strategies.base import (
    ImputationStrategyEnum,
    STRATEGY_REGISTRY,
)
from .partial import (
    imputation_compute_partial,
    impute_locally,
    impute_and_train_lr,
    _impute_and_train_lr_core, 
    validate_partial,
    get_unique_event_times_partial,
)


def _pool_km_event_tables(lr_partials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pool node-level KM event tables by summing counts per time.
    D2T events correspond to the 'observed' column in your KM event table.
    """
    tables = []
    for p in lr_partials:
        km = (p or {}).get("km") or {}
        km_json = km.get("km_event_table")
        if not km_json:
            continue
        tables.append(pd.read_json(StringIO(km_json)))

    if not tables:
        return {"pooled_km_event_table": None, "d2t_events": None}

    pooled = pd.concat(tables, ignore_index=True)

    # Identify time column: everything except the known count columns
    count_cols = {"removed", "observed", "interval", "censored", "at_risk"}
    time_cols = [c for c in pooled.columns if c not in count_cols]
    if not time_cols:
        raise ValueError(f"Could not infer KM time column from columns={list(pooled.columns)}")
    time_col = time_cols[0]

    agg = (
        pooled.groupby(time_col, as_index=False)[["removed", "observed", "interval", "censored"]]
        .sum()
        .sort_values(time_col)
    )

    # recompute at_risk consistently with node definition
    agg["at_risk"] = agg["removed"].iloc[::-1].cumsum().iloc[::-1]

    # Meta outcome requested: D2T event counts over time = 'observed'
    d2t_events = agg[[time_col, "observed"]].rename(columns={"observed": "d2t_events"})

    return {
        "km_time_col": time_col,
        "pooled_km_event_table": agg.to_dict(orient="records"),
        "d2t_events": d2t_events.to_dict(orient="records"),
    }


# -------------------
# Vantage6 entrypoint
# -------------------
@algorithm_client
def main(
    client: AlgorithmClient,
    *,
    columns: List[str],
    predictors: List[str],
    outcome: str,
    organizations: Optional[List[int]] = None,
    n_local_iterations: int = 50,
    model_name: Optional[str] = None,
    run_validation: bool = True,
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER,
    run_km: bool = True,
    km_noise_type=None,
    km_snr: Optional[float] = None,
    km_random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Orchestrate validation -> imputation metrics -> (KM unique times) -> LR(+KM) across nodes.

    Returns validation summaries, global imputation metrics, per-node LR results,
    aggregated global LR parameters, and pooled KM / D2T-event outcome.
    """
    org_ids = organizations or [org["id"] for org in client.organization.list()]
    results: Dict[str, Any] = {"organizations": org_ids}

    # 1) Validate locally on each node (optional)
    if run_validation:
        val_task = client.task.create(
            input_={"method": "validate_partial", "kwargs": {"model_name": model_name}},
            organizations=org_ids,
        )
        results["validation"] = client.wait_for_results(task_id=val_task["id"])

    # 2) Imputation metrics per node
    imp_task = client.task.create(
        input_={
            "method": "imputation_compute_partial",
            "kwargs": {
                "columns": columns,
                "imputation_strategy": imputation_strategy,
            },
        },
        organizations=org_ids,
    )
    node_metrics = client.wait_for_results(task_id=imp_task["id"])

    imputer_cls = STRATEGY_REGISTRY[imputation_strategy]
    imputer = imputer_cls()
    global_metrics = imputer.aggregate(node_metrics=node_metrics, columns=columns)
    results["imputation_metrics"] = global_metrics

    # 2.5) KM: compute GLOBAL unique_event_times (union across nodes)
    global_unique_event_times: Optional[List[float]] = None
    if run_km:
        times_task = client.task.create(
            input_={
                "method": "get_unique_event_times_partial",
                "kwargs": {
                    "noise_type": km_noise_type,
                    "snr": km_snr,
                    "random_seed": km_random_seed,
                },
            },
            organizations=org_ids,
        )
        node_times = client.wait_for_results(task_id=times_task["id"])

        merged: Set[float] = set()
        for tlist in node_times:
            merged.update(tlist)

        global_unique_event_times = sorted(merged)
        results["global_unique_event_times"] = global_unique_event_times

    # 3) Train LR with imputed data on each node (+KM on same global time grid)
    lr_task = client.task.create(
        input_={
            "method": "impute_and_train_lr",
            "kwargs": {
                "global_metrics": global_metrics,
                "predictors": predictors,
                "outcome": outcome,
                "n_local_iterations": n_local_iterations,
                "imputation_strategy": imputation_strategy,

                # KM args
                "run_km": run_km,
                "unique_event_times": global_unique_event_times,
                "km_noise_type": km_noise_type,
                "km_snr": km_snr,
                "km_random_seed": km_random_seed,
            },
        },
        organizations=org_ids,
    )
    lr_partials = client.wait_for_results(task_id=lr_task["id"])
    results["lr_partials"] = lr_partials
    results["lr_model_attributes"] = _aggregate_lr_models(lr_partials)

    # 4) Meta outcome: pooled KM + D2T events (observed)
    if run_km:
        km_pooled = _pool_km_event_tables(lr_partials)
        results["km_pooled"] = km_pooled
        results["meta_outcome_d2t_events"] = km_pooled["d2t_events"]

    return results


def _aggregate_lr_models(partials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple FedAvg over coef_ and intercept_, weighted by local sample size."""
    if not partials:
        return {}
    total = sum(p.get("size", 0) for p in partials if isinstance(p, dict))
    if total == 0:
        return partials[0].get("model_attributes", {})

    first = partials[0]["model_attributes"]
    coef_sum = np.zeros_like(np.array(first["coef_"], dtype=float))
    inter_sum = np.zeros_like(np.array(first["intercept_"], dtype=float))

    for p in partials:
        w = p.get("size", 0)
        ma = p["model_attributes"]
        coef_sum += np.array(ma["coef_"], dtype=float) * w
        inter_sum += np.array(ma["intercept_"], dtype=float) * w

    avg_coef = (coef_sum / total).tolist()
    avg_inter = (inter_sum / total).tolist()

    return {
        "coef_": avg_coef,
        "intercept_": avg_inter,
        "classes_": first["classes_"],
    }


# -----------------------
# Local/mock helper (kept)
# -----------------------
def run_pipeline(
    df: pd.DataFrame,
    *,
    imputation_columns: List[str],
    predictors: List[str],
    outcome: str,
    n_local_iterations: int = 50,
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER,
) -> Dict[str, Any]:
    """
    Local mock: compute global metrics from a single dataframe,
    then run core once.
    """
    imputer_cls = STRATEGY_REGISTRY[imputation_strategy]
    imputer = imputer_cls()
    node_metric = imputer.compute(df, imputation_columns).to_dict()
    global_metrics = imputer.aggregate([node_metric], imputation_columns)

    lr_result = _impute_and_train_lr_core(
        df,  # pass raw df; core imputes
        global_metrics=global_metrics,
        predictors=predictors,
        outcome=outcome,
        n_local_iterations=n_local_iterations,
        imputation_strategy=imputation_strategy,
        run_km=True,
        unique_event_times=None,
    )

    return {
        "imputation_metrics": global_metrics,
        "lr_model_attributes": lr_result.get("model_attributes"),
        "training_size": lr_result.get("size"),
        "km": lr_result.get("km", {}),
    }
