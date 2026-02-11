"""
Central orchestrator for the STRATA-FIT meta-algorithm.
Exposes a Vantage6 entrypoint and keeps a local helper for mock testing.
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient

from strata_fit_v6_imputation_py.imputation_strategies.base import (
    ImputationStrategyEnum,
    STRATEGY_REGISTRY,
)
from .partial import (
    _coerce_strategy,
    imputation_compute_partial,
    impute_locally,
    impute_and_train_lr,
    _impute_and_train_lr_core,
    validate_partial,
)


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
) -> Dict[str, Any]:
    """
    Orchestrate validation -> imputation -> LR training across nodes.

    Returns validation summaries, global imputation metrics, per-node LR results,
    and aggregated global LR parameters.
    """
    org_ids = organizations or [org["id"] for org in client.organization.list()]
    strategy = _coerce_strategy(imputation_strategy)

    results: Dict[str, Any] = {"organizations": org_ids}

    # 1) Validate locally on each node (optional)
    if run_validation:
        val_task = client.task.create(
            input_={"method": "validate_partial", "kwargs": {"model_name": model_name}},
            organizations=org_ids,
        )
        validation = client.wait_for_results(task_id=val_task["id"])
        results["validation"] = validation

    # 2) Imputation metrics per node
    imp_task = client.task.create(
        input_={
            "method": "imputation_compute_partial",
            "kwargs": {
                "columns": columns,
                "imputation_strategy": strategy,
            },
        },
        organizations=org_ids,
    )
    node_metrics = client.wait_for_results(task_id=imp_task["id"])

    imputer_cls = STRATEGY_REGISTRY[strategy]
    imputer = imputer_cls()
    global_metrics = imputer.aggregate(node_metrics=node_metrics, columns=columns)
    results["imputation_metrics"] = global_metrics

    # 3) Train LR with imputed data on each node
    lr_task = client.task.create(
        input_={
            "method": "impute_and_train_lr",
            "kwargs": {
                "global_metrics": global_metrics,
                "predictors": predictors,
                "outcome": outcome,
                "n_local_iterations": n_local_iterations,
                "imputation_strategy": strategy,
            },
        },
        organizations=org_ids,
    )
    lr_partials = client.wait_for_results(task_id=lr_task["id"])
    results["lr_partials"] = lr_partials
    results["lr_model_attributes"] = _aggregate_lr_models(lr_partials)

    return results


def _aggregate_lr_models(partials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple FedAvg over coef_ and intercept_, weighted by local sample size."""
    if not partials:
        return {}
    total = sum(p.get("size", 0) for p in partials)
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
    impute, and train LR once (used in tests).
    """
    strategy = _coerce_strategy(imputation_strategy)
    imputer_cls = STRATEGY_REGISTRY[strategy]
    imputer = imputer_cls()
    node_metric = imputer.compute(df, imputation_columns).to_dict()
    global_metrics = imputer.aggregate([node_metric], imputation_columns)
    imputed_df = impute_locally(
        df,
        global_metrics,
        imputation_strategy=strategy,
    )

    init_attrs = {
        "coef_": [[0.0 for _ in predictors]],
        "intercept_": [0.0],
        "classes_": [0, 1],
    }
    lr_result = _impute_and_train_lr_core(
        imputed_df,
        global_metrics=global_metrics,
        predictors=predictors,
        outcome=outcome,
        n_local_iterations=n_local_iterations,
        imputation_strategy=imputation_strategy,
    )

    return {
        "imputation_metrics": global_metrics,
        "lr_model_attributes": lr_result["model_attributes"],
        "training_size": lr_result["size"],
    }
