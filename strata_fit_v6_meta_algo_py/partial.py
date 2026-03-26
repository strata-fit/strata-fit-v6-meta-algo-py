from typing import Any, Dict, List, Optional

import pandas as pd
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client, data
from v6_federated_core import MethodContext, dispatch_registered_method, to_v6_result

from .methods import METHOD_REGISTRY


@data(1)
def validate_partial(
    df: pd.DataFrame,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "validate_partial",
        {"model_name": model_name},
        context=MethodContext(method="validate_partial", meta={"df": df}),
    )
    return to_v6_result(envelope)


@data(1)
def imputation_compute_partial(
    df: pd.DataFrame,
    columns: List[str],
    imputation_strategy: str = "mean",
) -> Dict[str, Any]:
    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "imputation_compute_partial",
        {
            "columns": columns,
            "imputation_strategy": imputation_strategy,
        },
        context=MethodContext(method="imputation_compute_partial", meta={"df": df}),
    )
    return to_v6_result(envelope)


@data(1)
def impute_and_train_sklearn_linear(
    df: pd.DataFrame,
    global_metrics: Dict[str, Any],
    predictors: List[str],
    outcome: str,
    n_local_iterations: int = 50,
    model_class: Any = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    imputation_strategy: str = "mean",
) -> Dict[str, Any]:
    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "impute_and_train_sklearn_linear",
        {
            "global_metrics": global_metrics,
            "predictors": predictors,
            "outcome": outcome,
            "n_local_iterations": n_local_iterations,
            "model_class": model_class,
            "model_kwargs": model_kwargs or {},
            "imputation_strategy": imputation_strategy,
        },
        context=MethodContext(method="impute_and_train_sklearn_linear", meta={"df": df}),
    )
    return to_v6_result(envelope)


@data(1)
@algorithm_client
def cox_get_unique_event_times_imputed(
    client: AlgorithmClient,
    df: pd.DataFrame,
    time_col: str,
    outcome_col: str,
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    minimum_events: int = 10,
    preprocess_raw_data: bool = False,
) -> Dict[str, Any]:
    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "cox_get_unique_event_times_imputed",
        {
            "time_col": time_col,
            "outcome_col": outcome_col,
            "global_metrics": global_metrics,
            "imputation_strategy": imputation_strategy,
            "minimum_events": minimum_events,
            "preprocess_raw_data": preprocess_raw_data,
        },
        context=MethodContext(
            method="cox_get_unique_event_times_imputed",
            meta={"df": df, "client": client},
        ),
    )
    return to_v6_result(envelope)


@data(1)
def cox_compute_summed_z_imputed(
    df: pd.DataFrame,
    outcome_col: str,
    expl_vars: List[str],
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = False,
) -> Dict[str, Any]:
    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "cox_compute_summed_z_imputed",
        {
            "outcome_col": outcome_col,
            "expl_vars": expl_vars,
            "global_metrics": global_metrics,
            "imputation_strategy": imputation_strategy,
            "preprocess_raw_data": preprocess_raw_data,
        },
        context=MethodContext(method="cox_compute_summed_z_imputed", meta={"df": df}),
    )
    return to_v6_result(envelope)


@data(1)
def cox_perform_iteration_imputed(
    df: pd.DataFrame,
    time_col: str,
    expl_vars: List[str],
    beta: List[float],
    unique_time_events: List[float],
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = False,
) -> Dict[str, Any]:
    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "cox_perform_iteration_imputed",
        {
            "time_col": time_col,
            "expl_vars": expl_vars,
            "beta": beta,
            "unique_time_events": unique_time_events,
            "global_metrics": global_metrics,
            "imputation_strategy": imputation_strategy,
            "preprocess_raw_data": preprocess_raw_data,
        },
        context=MethodContext(method="cox_perform_iteration_imputed", meta={"df": df}),
    )
    return to_v6_result(envelope)


@data(1)
def km_get_unique_event_times_imputed(
    df: pd.DataFrame,
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = True,
) -> Dict[str, Any]:
    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "km_get_unique_event_times_imputed",
        {
            "global_metrics": global_metrics,
            "imputation_strategy": imputation_strategy,
            "preprocess_raw_data": preprocess_raw_data,
        },
        context=MethodContext(method="km_get_unique_event_times_imputed", meta={"df": df}),
    )
    return to_v6_result(envelope)


@data(1)
def km_get_event_table_imputed(
    df: pd.DataFrame,
    unique_event_times: List[float],
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = True,
) -> Dict[str, Any]:
    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "km_get_event_table_imputed",
        {
            "unique_event_times": unique_event_times,
            "global_metrics": global_metrics,
            "imputation_strategy": imputation_strategy,
            "preprocess_raw_data": preprocess_raw_data,
        },
        context=MethodContext(method="km_get_event_table_imputed", meta={"df": df}),
    )
    return to_v6_result(envelope)
