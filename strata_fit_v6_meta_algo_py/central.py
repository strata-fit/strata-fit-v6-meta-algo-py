from typing import Any, Dict, List, Optional

from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client
from v6_federated_core import MethodContext, dispatch_registered_method, to_v6_result

from .contracts import FinalModelEnum
from .methods import (
    METHOD_REGISTRY,
    build_min_organization_policies,
    build_policy_context,
)


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


@algorithm_client
def main(
    client: AlgorithmClient,
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
) -> Dict[str, Any]:
    """
    Meta orchestrator:
      validation -> imputation metrics -> final federated model.

    `final_model` supported values:
      - `sklearn_linear`
      - `cox`
    """
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

    resolved_org_ids = organizations or [org["id"] for org in client.organization.list()]

    method_context = MethodContext(
        method="main",
        organization_ids=resolved_org_ids,
        meta={"client": client},
    )
    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "main",
        {
            "columns": columns,
            "organizations": organizations,
            "model_name": model_name,
            "run_validation": run_validation,
            "imputation_strategy": imputation_strategy,
            "final_model": resolved_final_model,
            "final_model_config": resolved_final_config,
        },
        context=method_context,
        policies=build_min_organization_policies(),
        policy_context=build_policy_context("main", resolved_org_ids),
    )
    return to_v6_result(envelope)
