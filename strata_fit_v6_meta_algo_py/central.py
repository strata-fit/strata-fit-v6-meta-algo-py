from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .client import AlgorithmProxyClient
from .contracts import FinalModelEnum
from .io import normalize_payload, write_output
from .local_client import run_local_main
from .runtime import run_context
from .service import run_central_method


def _build_legacy_final_model_config(
    final_model: FinalModelEnum,
    *,
    predictors: Optional[List[str]],
    outcome: Optional[str],
    n_local_iterations: int,
    model_class: Any,
    model_kwargs: Optional[Dict[str, Any]],
    time_col: Optional[str],
    outcome_col: Optional[str],
    expl_vars: Optional[List[str]],
    max_iterations: int,
    tolerance: float,
) -> Dict[str, Any]:
    if final_model == FinalModelEnum.SKLEARN_LINEAR:
        if not predictors or not outcome:
            raise ValueError(
                "For final_model='sklearn_linear', provide predictors and outcome "
                "or pass final_model_config explicitly."
            )
        return {
            "predictors": predictors,
            "outcome": outcome,
            "n_local_iterations": n_local_iterations,
            "model_class": model_class,
            "model_kwargs": model_kwargs or {},
        }

    if final_model == FinalModelEnum.COX:
        resolved_outcome_col = outcome_col or outcome
        if not time_col or not resolved_outcome_col or not expl_vars:
            raise ValueError(
                "For final_model='cox', provide time_col, outcome_col (or outcome), and expl_vars "
                "or pass final_model_config explicitly."
            )
        return {
            "time_col": time_col,
            "outcome_col": resolved_outcome_col,
            "expl_vars": expl_vars,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
        }

    if final_model == FinalModelEnum.KM:
        return {}

    raise ValueError(f"Unsupported final_model for legacy config: {final_model}")


@run_context(
    output_uris="output_path",
    named_arguments=[
        "columns",
        "organizations",
        "model_name",
        "run_validation",
        "imputation_strategy",
        "final_model",
        "final_model_config",
        "predictors",
        "outcome",
        "n_local_iterations",
        "model_class",
        "model_kwargs",
        "time_col",
        "outcome_col",
        "expl_vars",
        "max_iterations",
        "tolerance",
    ],
)
def main(
    *,
    columns: List[str],
    organizations: Optional[List[int]] = None,
    model_name: Optional[str] = None,
    run_validation: bool = True,
    imputation_strategy: str = "mean",
    final_model: str = "sklearn_linear",
    final_model_config: Optional[Dict[str, Any]] = None,
    predictors: Optional[List[str]] = None,
    outcome: Optional[str] = None,
    n_local_iterations: int = 50,
    model_class: Any = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    time_col: Optional[str] = None,
    outcome_col: Optional[str] = None,
    expl_vars: Optional[List[str]] = None,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
    output_path: str | Path | None = None,
    client: Any = None,
) -> Dict[str, Any]:
    resolved_final_model = FinalModelEnum(final_model)
    resolved_final_config = final_model_config or _build_legacy_final_model_config(
        resolved_final_model,
        predictors=predictors,
        outcome=outcome,
        n_local_iterations=n_local_iterations,
        model_class=model_class,
        model_kwargs=model_kwargs,
        time_col=time_col,
        outcome_col=outcome_col,
        expl_vars=expl_vars,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    resolved_client = client or AlgorithmProxyClient.from_env()
    resolved_org_ids = organizations or [
        org["id"] for org in resolved_client.organization.list()
    ]
    result = normalize_payload(
        run_central_method(
        raw_input={
            "columns": columns,
            "organizations": organizations,
            "model_name": model_name,
            "run_validation": run_validation,
            "imputation_strategy": imputation_strategy,
            "final_model": resolved_final_model,
            "final_model_config": resolved_final_config,
        },
        client=resolved_client,
        organization_ids=resolved_org_ids,
        )
    )
    write_output(output_path, result)
    return result


def run_local_meta_algorithm(
    datasets: list[pd.DataFrame],
    **kwargs: Any,
) -> Dict[str, Any]:
    return run_local_main(datasets, **kwargs)
