from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .io import normalize_payload, write_output
from .runtime import run_context
from .service import run_partial_method


def _load_dataframe(dataset_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(dataset_path))


def validate_partial_frame(
    df: pd.DataFrame,
    *,
    model_name: Optional[str] = None,
    client: Any = None,
) -> Dict[str, Any]:
    return run_partial_method(
        "validate_partial",
        df=df,
        raw_input={"model_name": model_name},
        client=client,
    )


def imputation_compute_partial_frame(
    df: pd.DataFrame,
    *,
    columns: List[str],
    imputation_strategy: str = "mean",
    client: Any = None,
) -> Dict[str, Any]:
    return run_partial_method(
        "imputation_compute_partial",
        df=df,
        raw_input={
            "columns": columns,
            "imputation_strategy": imputation_strategy,
        },
        client=client,
    )


def impute_and_train_sklearn_linear_frame(
    df: pd.DataFrame,
    *,
    global_metrics: Dict[str, Any],
    predictors: List[str],
    outcome: str,
    n_local_iterations: int = 50,
    model_class: Any = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    imputation_strategy: str = "mean",
    client: Any = None,
) -> Dict[str, Any]:
    return run_partial_method(
        "impute_and_train_sklearn_linear",
        df=df,
        raw_input={
            "global_metrics": global_metrics,
            "predictors": predictors,
            "outcome": outcome,
            "n_local_iterations": n_local_iterations,
            "model_class": model_class,
            "model_kwargs": model_kwargs or {},
            "imputation_strategy": imputation_strategy,
        },
        client=client,
    )


def cox_get_unique_event_times_imputed_frame(
    df: pd.DataFrame,
    *,
    time_col: str,
    outcome_col: str,
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    minimum_events: int = 10,
    preprocess_raw_data: bool = False,
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    client: Any = None,
) -> Dict[str, Any]:
    return run_partial_method(
        "cox_get_unique_event_times_imputed",
        df=df,
        raw_input={
            "time_col": time_col,
            "outcome_col": outcome_col,
            "global_metrics": global_metrics,
            "imputation_strategy": imputation_strategy,
            "minimum_events": minimum_events,
            "preprocess_raw_data": preprocess_raw_data,
            "cohort": cohort or {},
            "event_definition": event_definition,
        },
        client=client,
    )


def cox_compute_summed_z_imputed_frame(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    expl_vars: List[str],
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = False,
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    client: Any = None,
) -> Dict[str, Any]:
    return run_partial_method(
        "cox_compute_summed_z_imputed",
        df=df,
        raw_input={
            "outcome_col": outcome_col,
            "expl_vars": expl_vars,
            "global_metrics": global_metrics,
            "imputation_strategy": imputation_strategy,
            "preprocess_raw_data": preprocess_raw_data,
            "cohort": cohort or {},
            "event_definition": event_definition,
        },
        client=client,
    )


def cox_perform_iteration_imputed_frame(
    df: pd.DataFrame,
    *,
    time_col: str,
    expl_vars: List[str],
    beta: List[float],
    unique_time_events: List[float],
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = False,
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    client: Any = None,
) -> Dict[str, Any]:
    return run_partial_method(
        "cox_perform_iteration_imputed",
        df=df,
        raw_input={
            "time_col": time_col,
            "expl_vars": expl_vars,
            "beta": beta,
            "unique_time_events": unique_time_events,
            "global_metrics": global_metrics,
            "imputation_strategy": imputation_strategy,
            "preprocess_raw_data": preprocess_raw_data,
            "cohort": cohort or {},
            "event_definition": event_definition,
        },
        client=client,
    )


def km_get_unique_event_times_imputed_frame(
    df: pd.DataFrame,
    *,
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = True,
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    client: Any = None,
) -> Dict[str, Any]:
    return run_partial_method(
        "km_get_unique_event_times_imputed",
        df=df,
        raw_input={
            "global_metrics": global_metrics,
            "imputation_strategy": imputation_strategy,
            "preprocess_raw_data": preprocess_raw_data,
            "cohort": cohort or {},
            "event_definition": event_definition,
        },
        client=client,
    )


def km_get_event_table_imputed_frame(
    df: pd.DataFrame,
    *,
    unique_event_times: List[float],
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = True,
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    client: Any = None,
) -> Dict[str, Any]:
    return run_partial_method(
        "km_get_event_table_imputed",
        df=df,
        raw_input={
            "unique_event_times": unique_event_times,
            "global_metrics": global_metrics,
            "imputation_strategy": imputation_strategy,
            "preprocess_raw_data": preprocess_raw_data,
            "cohort": cohort or {},
            "event_definition": event_definition,
        },
        client=client,
    )


def prevalence_by_year_imputed_frame(
    df: pd.DataFrame,
    *,
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    client: Any = None,
) -> Dict[str, Any]:
    return run_partial_method(
        "prevalence_by_year_imputed",
        df=df,
        raw_input={
            "global_metrics": global_metrics,
            "imputation_strategy": imputation_strategy,
            "cohort": cohort or {},
            "event_definition": event_definition,
        },
        client=client,
    )


def d2t_characteristics_frame(
    df: pd.DataFrame,
    *,
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    client: Any = None,
) -> Dict[str, Any]:
    return run_partial_method(
        "d2t_characteristics",
        df=df,
        raw_input={
            "global_metrics": global_metrics,
            "imputation_strategy": imputation_strategy,
            "cohort": cohort or {},
            "event_definition": event_definition,
        },
        client=client,
    )


def cox_risk_score_range_imputed_frame(
    df: pd.DataFrame,
    *,
    time_col: str,
    outcome_col: str,
    expl_vars: List[str],
    beta: List[float],
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = False,
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    client: Any = None,
) -> Dict[str, Any]:
    return run_partial_method(
        "cox_risk_score_range_imputed",
        df=df,
        raw_input={
            "time_col": time_col,
            "outcome_col": outcome_col,
            "expl_vars": expl_vars,
            "beta": beta,
            "global_metrics": global_metrics,
            "imputation_strategy": imputation_strategy,
            "preprocess_raw_data": preprocess_raw_data,
            "cohort": cohort or {},
            "event_definition": event_definition,
        },
        client=client,
    )


def cox_risk_score_histogram_imputed_frame(
    df: pd.DataFrame,
    *,
    time_col: str,
    outcome_col: str,
    expl_vars: List[str],
    beta: List[float],
    bin_edges: List[float],
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = False,
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    client: Any = None,
) -> Dict[str, Any]:
    return run_partial_method(
        "cox_risk_score_histogram_imputed",
        df=df,
        raw_input={
            "time_col": time_col,
            "outcome_col": outcome_col,
            "expl_vars": expl_vars,
            "beta": beta,
            "bin_edges": bin_edges,
            "global_metrics": global_metrics,
            "imputation_strategy": imputation_strategy,
            "preprocess_raw_data": preprocess_raw_data,
            "cohort": cohort or {},
            "event_definition": event_definition,
        },
        client=client,
    )


def cox_risk_group_summary_imputed_frame(
    df: pd.DataFrame,
    *,
    time_col: str,
    outcome_col: str,
    expl_vars: List[str],
    beta: List[float],
    cutoffs: List[float],
    horizons_months: List[int],
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = False,
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    client: Any = None,
) -> Dict[str, Any]:
    return run_partial_method(
        "cox_risk_group_summary_imputed",
        df=df,
        raw_input={
            "time_col": time_col,
            "outcome_col": outcome_col,
            "expl_vars": expl_vars,
            "beta": beta,
            "cutoffs": cutoffs,
            "horizons_months": horizons_months,
            "global_metrics": global_metrics,
            "imputation_strategy": imputation_strategy,
            "preprocess_raw_data": preprocess_raw_data,
            "cohort": cohort or {},
            "event_definition": event_definition,
        },
        client=client,
    )


@run_context(input_uris="dataset_path", output_uris="output_path", named_arguments=["model_name"])
def validate_partial(
    dataset_path: str | Path,
    model_name: Optional[str] = None,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    result = normalize_payload(
        validate_partial_frame(_load_dataframe(dataset_path), model_name=model_name)
    )
    write_output(output_path, result)
    return result


@run_context(
    input_uris="dataset_path",
    output_uris="output_path",
    named_arguments=["columns", "imputation_strategy"],
)
def imputation_compute_partial(
    dataset_path: str | Path,
    columns: List[str],
    imputation_strategy: str = "mean",
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    result = normalize_payload(
        imputation_compute_partial_frame(
            _load_dataframe(dataset_path),
            columns=columns,
            imputation_strategy=imputation_strategy,
        )
    )
    write_output(output_path, result)
    return result


@run_context(
    input_uris="dataset_path",
    output_uris="output_path",
    named_arguments=[
        "global_metrics",
        "predictors",
        "outcome",
        "n_local_iterations",
        "model_class",
        "model_kwargs",
        "imputation_strategy",
    ],
)
def impute_and_train_sklearn_linear(
    dataset_path: str | Path,
    global_metrics: Dict[str, Any],
    predictors: List[str],
    outcome: str,
    n_local_iterations: int = 50,
    model_class: Any = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    imputation_strategy: str = "mean",
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    result = normalize_payload(
        impute_and_train_sklearn_linear_frame(
            _load_dataframe(dataset_path),
            global_metrics=global_metrics,
            predictors=predictors,
            outcome=outcome,
            n_local_iterations=n_local_iterations,
            model_class=model_class,
            model_kwargs=model_kwargs,
            imputation_strategy=imputation_strategy,
        )
    )
    write_output(output_path, result)
    return result


@run_context(
    input_uris="dataset_path",
    output_uris="output_path",
    named_arguments=[
        "time_col",
        "outcome_col",
        "global_metrics",
        "imputation_strategy",
        "minimum_events",
        "preprocess_raw_data",
        "cohort",
        "event_definition",
    ],
)
def cox_get_unique_event_times_imputed(
    dataset_path: str | Path,
    time_col: str,
    outcome_col: str,
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    minimum_events: int = 10,
    preprocess_raw_data: bool = False,
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    result = normalize_payload(
        cox_get_unique_event_times_imputed_frame(
            _load_dataframe(dataset_path),
            time_col=time_col,
            outcome_col=outcome_col,
            global_metrics=global_metrics,
            imputation_strategy=imputation_strategy,
            minimum_events=minimum_events,
            preprocess_raw_data=preprocess_raw_data,
            cohort=cohort,
            event_definition=event_definition,
        )
    )
    write_output(output_path, result)
    return result


@run_context(
    input_uris="dataset_path",
    output_uris="output_path",
    named_arguments=[
        "outcome_col",
        "expl_vars",
        "global_metrics",
        "imputation_strategy",
        "preprocess_raw_data",
        "cohort",
        "event_definition",
    ],
)
def cox_compute_summed_z_imputed(
    dataset_path: str | Path,
    outcome_col: str,
    expl_vars: List[str],
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = False,
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    result = normalize_payload(
        cox_compute_summed_z_imputed_frame(
            _load_dataframe(dataset_path),
            outcome_col=outcome_col,
            expl_vars=expl_vars,
            global_metrics=global_metrics,
            imputation_strategy=imputation_strategy,
            preprocess_raw_data=preprocess_raw_data,
            cohort=cohort,
            event_definition=event_definition,
        )
    )
    write_output(output_path, result)
    return result


@run_context(
    input_uris="dataset_path",
    output_uris="output_path",
    named_arguments=[
        "time_col",
        "expl_vars",
        "beta",
        "unique_time_events",
        "global_metrics",
        "imputation_strategy",
        "preprocess_raw_data",
        "cohort",
        "event_definition",
    ],
)
def cox_perform_iteration_imputed(
    dataset_path: str | Path,
    time_col: str,
    expl_vars: List[str],
    beta: List[float],
    unique_time_events: List[float],
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = False,
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    result = normalize_payload(
        cox_perform_iteration_imputed_frame(
            _load_dataframe(dataset_path),
            time_col=time_col,
            expl_vars=expl_vars,
            beta=beta,
            unique_time_events=unique_time_events,
            global_metrics=global_metrics,
            imputation_strategy=imputation_strategy,
            preprocess_raw_data=preprocess_raw_data,
            cohort=cohort,
            event_definition=event_definition,
        )
    )
    write_output(output_path, result)
    return result


@run_context(
    input_uris="dataset_path",
    output_uris="output_path",
    named_arguments=[
        "global_metrics",
        "imputation_strategy",
        "preprocess_raw_data",
        "cohort",
        "event_definition",
    ],
)
def km_get_unique_event_times_imputed(
    dataset_path: str | Path,
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = True,
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    result = normalize_payload(
        km_get_unique_event_times_imputed_frame(
            _load_dataframe(dataset_path),
            global_metrics=global_metrics,
            imputation_strategy=imputation_strategy,
            preprocess_raw_data=preprocess_raw_data,
            cohort=cohort,
            event_definition=event_definition,
        )
    )
    write_output(output_path, result)
    return result


@run_context(
    input_uris="dataset_path",
    output_uris="output_path",
    named_arguments=[
        "unique_event_times",
        "global_metrics",
        "imputation_strategy",
        "preprocess_raw_data",
        "cohort",
        "event_definition",
    ],
)
def km_get_event_table_imputed(
    dataset_path: str | Path,
    unique_event_times: List[float],
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = True,
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    result = normalize_payload(
        km_get_event_table_imputed_frame(
            _load_dataframe(dataset_path),
            unique_event_times=unique_event_times,
            global_metrics=global_metrics,
            imputation_strategy=imputation_strategy,
            preprocess_raw_data=preprocess_raw_data,
            cohort=cohort,
            event_definition=event_definition,
        )
    )
    write_output(output_path, result)
    return result


@run_context(
    input_uris="dataset_path",
    output_uris="output_path",
    named_arguments=["global_metrics", "imputation_strategy", "cohort", "event_definition"],
)
def prevalence_by_year_imputed(
    dataset_path: str | Path,
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    result = normalize_payload(
        prevalence_by_year_imputed_frame(
            _load_dataframe(dataset_path),
            global_metrics=global_metrics,
            imputation_strategy=imputation_strategy,
            cohort=cohort,
            event_definition=event_definition,
        )
    )
    write_output(output_path, result)
    return result


@run_context(
    input_uris="dataset_path",
    output_uris="output_path",
    named_arguments=["global_metrics", "imputation_strategy", "cohort", "event_definition"],
)
def d2t_characteristics(
    dataset_path: str | Path,
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    result = normalize_payload(
        d2t_characteristics_frame(
            _load_dataframe(dataset_path),
            global_metrics=global_metrics,
            imputation_strategy=imputation_strategy,
            cohort=cohort,
            event_definition=event_definition,
        )
    )
    write_output(output_path, result)
    return result


@run_context(
    input_uris="dataset_path",
    output_uris="output_path",
    named_arguments=[
        "time_col",
        "outcome_col",
        "expl_vars",
        "beta",
        "global_metrics",
        "imputation_strategy",
        "preprocess_raw_data",
        "cohort",
        "event_definition",
    ],
)
def cox_risk_score_range_imputed(
    dataset_path: str | Path,
    time_col: str,
    outcome_col: str,
    expl_vars: List[str],
    beta: List[float],
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = False,
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    result = normalize_payload(
        cox_risk_score_range_imputed_frame(
            _load_dataframe(dataset_path),
            time_col=time_col,
            outcome_col=outcome_col,
            expl_vars=expl_vars,
            beta=beta,
            global_metrics=global_metrics,
            imputation_strategy=imputation_strategy,
            preprocess_raw_data=preprocess_raw_data,
            cohort=cohort,
            event_definition=event_definition,
        )
    )
    write_output(output_path, result)
    return result


@run_context(
    input_uris="dataset_path",
    output_uris="output_path",
    named_arguments=[
        "time_col",
        "outcome_col",
        "expl_vars",
        "beta",
        "bin_edges",
        "global_metrics",
        "imputation_strategy",
        "preprocess_raw_data",
        "cohort",
        "event_definition",
    ],
)
def cox_risk_score_histogram_imputed(
    dataset_path: str | Path,
    time_col: str,
    outcome_col: str,
    expl_vars: List[str],
    beta: List[float],
    bin_edges: List[float],
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = False,
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    result = normalize_payload(
        cox_risk_score_histogram_imputed_frame(
            _load_dataframe(dataset_path),
            time_col=time_col,
            outcome_col=outcome_col,
            expl_vars=expl_vars,
            beta=beta,
            bin_edges=bin_edges,
            global_metrics=global_metrics,
            imputation_strategy=imputation_strategy,
            preprocess_raw_data=preprocess_raw_data,
            cohort=cohort,
            event_definition=event_definition,
        )
    )
    write_output(output_path, result)
    return result


@run_context(
    input_uris="dataset_path",
    output_uris="output_path",
    named_arguments=[
        "time_col",
        "outcome_col",
        "expl_vars",
        "beta",
        "cutoffs",
        "horizons_months",
        "global_metrics",
        "imputation_strategy",
        "preprocess_raw_data",
        "cohort",
        "event_definition",
    ],
)
def cox_risk_group_summary_imputed(
    dataset_path: str | Path,
    time_col: str,
    outcome_col: str,
    expl_vars: List[str],
    beta: List[float],
    cutoffs: List[float],
    horizons_months: List[int],
    global_metrics: Dict[str, Any],
    imputation_strategy: str = "mean",
    preprocess_raw_data: bool = False,
    cohort: Dict[str, Any] | None = None,
    event_definition: str = "d2t_ra_v2026_selected_v1",
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    result = normalize_payload(
        cox_risk_group_summary_imputed_frame(
            _load_dataframe(dataset_path),
            time_col=time_col,
            outcome_col=outcome_col,
            expl_vars=expl_vars,
            beta=beta,
            cutoffs=cutoffs,
            horizons_months=horizons_months,
            global_metrics=global_metrics,
            imputation_strategy=imputation_strategy,
            preprocess_raw_data=preprocess_raw_data,
            cohort=cohort,
            event_definition=event_definition,
        )
    )
    write_output(output_path, result)
    return result
