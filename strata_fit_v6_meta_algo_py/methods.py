import math
import os
import warnings
from importlib import import_module
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import chi2, norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from v6_federated_core import (
    DataContractError,
    MethodContext,
    MethodRegistry,
    MethodSpec,
    MinOrganizationsPolicy,
    PolicyContext,
    PolicyScope,
    TaskRunner,
    WorkflowStepSpec,
)

from .contracts import (
    CoxComputeSummedZImputedInput,
    CoxComputeSummedZImputedOutput,
    CoxFinalConfig,
    CoxGetUniqueEventTimesImputedInput,
    CoxGetUniqueEventTimesImputedOutput,
    CoxPerformIterationImputedInput,
    CoxPerformIterationImputedOutput,
    CoxRiskGroupSummaryImputedInput,
    CoxRiskGroupSummaryImputedOutput,
    CoxRiskScoreHistogramImputedInput,
    CoxRiskScoreHistogramImputedOutput,
    CoxRiskScoreRangeImputedInput,
    CoxRiskScoreRangeImputedOutput,
    FinalModelEnum,
    ImputationComputePartialInput,
    ImputationComputePartialOutput,
    ImputeAndTrainSklearnLinearInput,
    ImputeAndTrainSklearnLinearOutput,
    KMFinalConfig,
    KMGetEventTableImputedInput,
    KMGetEventTableImputedOutput,
    KMGetUniqueEventTimesImputedInput,
    KMGetUniqueEventTimesImputedOutput,
    MetaCentralInput,
    MetaCentralOutput,
    PrevalenceByYearImputedInput,
    PrevalenceByYearImputedOutput,
    SklearnLinearFinalConfig,
    SurvivalBundleFinalConfig,
    ValidatePartialInput,
    ValidatePartialOutput,
)
from .cox import (
    ComputeSummedZInput as CoxComputeSummedZInput,
    GetUniqueEventTimesInput as CoxGetUniqueEventTimesInput,
    PerformIterationInput as CoxPerformIterationInput,
    compute_derivatives,
    compute_summed_z_handler as cox_compute_summed_z_handler,
    get_unique_event_times_handler as cox_get_unique_event_times_handler,
    perform_iteration_handler as cox_perform_iteration_handler,
)


def _bootstrap_validator_config_path() -> None:
    # The validator package reads CONFIG_PATH at import-time via Dynaconf. In
    # algorithm containers the working directory is not the project root, so
    # default "config/" lookup can fail unless we point to the installed package.
    if os.getenv("CONFIG_PATH"):
        return
    try:
        import config as validator_config_pkg
    except Exception:
        return

    config_dir = Path(validator_config_pkg.__file__).resolve().parent
    os.environ["CONFIG_PATH"] = str(config_dir)


_bootstrap_validator_config_path()

from .imputation import ImputationStrategyEnum, STRATEGY_REGISTRY
from .log import info, warn
from .preprocessing import (
    DEFAULT_EVENT_DEFINITION,
    DEFAULT_EVENT_INDICATOR_COLUMN,
    DEFAULT_INTERVAL_END_COLUMN,
    DEFAULT_INTERVAL_START_COLUMN,
    compute_d2t_prevalence_by_year,
    filter_dataframe_for_cohort,
    get_d2t_definition_config,
    list_supported_d2t_definitions,
    strata_fit_data_to_cox_input,
    strata_fit_data_to_km_input,
)
from .validator import load_data_models_from_settings, validate_csv

MAX_N_THRESHOLD_RETRIES = 3
LARGE_VALUE_WARNING_THRESHOLD = 10.0
MINIMUM_ORGANIZATIONS = 1


def _get_client(context: MethodContext) -> Any:
    client = context.meta.get("client")
    if client is None:
        raise RuntimeError("Method context is missing the AlgorithmClient")
    return client


def _get_dataframe(context: MethodContext) -> pd.DataFrame:
    df = context.meta.get("df")
    if df is None:
        raise RuntimeError("Method context is missing the dataframe")
    return df


def _coerce_strategy(strategy: Any) -> ImputationStrategyEnum:
    if isinstance(strategy, ImputationStrategyEnum):
        return strategy
    if isinstance(strategy, str):
        candidate = strategy.strip()
        for member in ImputationStrategyEnum:
            if candidate == member.value or candidate == member.name:
                return member
            if candidate.lower() == member.value.lower() or candidate.lower() == member.name.lower():
                return member
    raise ValueError(f"Unsupported imputation strategy: {strategy}")


def _resolve_organization_ids(
    client: Any,
    requested: Optional[Sequence[int]],
    context: MethodContext,
) -> List[int]:
    if requested:
        return list(dict.fromkeys(int(org_id) for org_id in requested))
    if context.organization_ids:
        return list(dict.fromkeys(int(org_id) for org_id in context.organization_ids))
    return [organization["id"] for organization in client.organization.list()]


def _impute_locally(
    df: pd.DataFrame,
    global_metrics: Dict[str, Any],
    strategy: ImputationStrategyEnum,
) -> pd.DataFrame:
    imputer_cls = STRATEGY_REGISTRY[strategy]
    imputer = imputer_cls()
    result = imputer.impute(df.copy(), global_metrics)
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, dict):
        return pd.DataFrame.from_dict(result)
    raise TypeError(
        "Imputation strategy returned unsupported type; expected pandas.DataFrame or dict"
    )


def _filter_and_impute_for_survival(
    df: pd.DataFrame,
    *,
    global_metrics: Dict[str, Any],
    strategy: ImputationStrategyEnum,
    cohort: Dict[str, Any] | None,
) -> pd.DataFrame:
    filtered = filter_dataframe_for_cohort(df, cohort)
    if filtered.empty:
        return filtered
    return _impute_locally(filtered, global_metrics, strategy)


def _resolve_linear_model_class(model_class: Any) -> type[BaseEstimator]:
    if model_class in (None, ""):
        return LogisticRegression

    if isinstance(model_class, str):
        if "." in model_class:
            module_name, class_name = model_class.rsplit(".", 1)
            cls = getattr(import_module(module_name), class_name)
        else:
            cls = getattr(import_module("sklearn.linear_model"), model_class)
    elif isinstance(model_class, type):
        cls = model_class
    else:
        raise ValueError("model_class must be a class or import string")

    if not issubclass(cls, BaseEstimator):
        raise ValueError("model_class must inherit from sklearn.base.BaseEstimator")
    return cls


def _filter_model_init_kwargs(
    model_class: type[BaseEstimator], candidate_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    sig = signature(model_class.__init__)
    accepts_var_kwargs = any(
        parameter.kind == Parameter.VAR_KEYWORD
        for parameter in sig.parameters.values()
    )
    if accepts_var_kwargs:
        return candidate_kwargs
    return {key: value for key, value in candidate_kwargs.items() if key in sig.parameters}


def _build_initial_model_attributes(
    model_class: type[BaseEstimator],
    predictors: List[str],
) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {
        "coef_": [[0.0 for _ in predictors]],
        "intercept_": [0.0],
    }
    if issubclass(model_class, ClassifierMixin):
        attrs["classes_"] = [0, 1]
    return attrs


def _initialize_model(
    model_class: type[BaseEstimator],
    model_attributes: Dict[str, Any],
    **model_init_kwargs: Any,
) -> BaseEstimator:
    model = model_class(**model_init_kwargs)
    for attribute, value in model_attributes.items():
        setattr(model, attribute, np.array(value))
    return model


def _train_local_sklearn_linear(
    df: pd.DataFrame,
    *,
    predictors: List[str],
    outcome: str,
    n_local_iterations: int,
    model_class: Any,
    model_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    model_cls = _resolve_linear_model_class(model_class)
    required_columns = list(dict.fromkeys([*predictors, outcome]))
    working = df.dropna(subset=required_columns, how="any")
    X = working[predictors].values
    y = working[outcome].values

    if X.shape[0] == 0:
        raise DataContractError("No rows available for sklearn-linear training after imputation")

    init_attrs = _build_initial_model_attributes(model_cls, predictors)
    candidate_kwargs = {
        "max_iter": n_local_iterations,
        "warm_start": True,
        **{key: value for key, value in model_kwargs.items() if value is not None},
    }
    init_kwargs = _filter_model_init_kwargs(model_cls, candidate_kwargs)
    model = _initialize_model(model_cls, init_attrs, **init_kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    attribute_keys = ["coef_", "intercept_"]
    if hasattr(model, "classes_"):
        attribute_keys.append("classes_")
    model_attributes = {
        key: np.asarray(getattr(model, key)).tolist()
        for key in attribute_keys
        if hasattr(model, key)
    }
    return {"model_attributes": model_attributes, "size": int(X.shape[0])}


def validate_partial_handler(
    data: ValidatePartialInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for validate_partial")

    df = _get_dataframe(context)
    models = load_data_models_from_settings()
    target = data.model_name or next(iter(models.keys()))
    model = models[target]

    _, errors = validate_csv(df, model)
    total_rows = len(df.index)
    total_errors = len(errors)
    return {
        "total_rows": total_rows,
        "total_errors": total_errors,
        "error_rate_per_row": (total_errors / total_rows) if total_rows else 0.0,
        "validation_passed": total_errors == 0,
    }


def imputation_compute_partial_handler(
    data: ImputationComputePartialInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for imputation_compute_partial")

    df = _get_dataframe(context)
    strategy = _coerce_strategy(data.imputation_strategy)
    imputer_cls = STRATEGY_REGISTRY[strategy]
    imputer = imputer_cls()
    result = imputer.compute(df, data.columns)
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    raise TypeError(
        "Imputation strategy returned unsupported type; expected dict or dataframe-like object with to_dict()"
    )


def impute_and_train_sklearn_linear_handler(
    data: ImputeAndTrainSklearnLinearInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for impute_and_train_sklearn_linear")

    df = _get_dataframe(context)
    strategy = _coerce_strategy(data.imputation_strategy)
    imputed = _impute_locally(df, data.global_metrics, strategy)
    return _train_local_sklearn_linear(
        imputed,
        predictors=data.predictors,
        outcome=data.outcome,
        n_local_iterations=data.n_local_iterations,
        model_class=data.model_class,
        model_kwargs=data.model_kwargs,
    )


def cox_get_unique_event_times_imputed_handler(
    data: CoxGetUniqueEventTimesImputedInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for cox_get_unique_event_times_imputed")

    df = _get_dataframe(context)
    client = context.meta.get("client")
    strategy = _coerce_strategy(data.imputation_strategy)
    imputed = _filter_and_impute_for_survival(
        df,
        global_metrics=data.global_metrics,
        strategy=strategy,
        cohort=data.cohort,
    )
    if data.preprocess_raw_data:
        imputed = strata_fit_data_to_cox_input(
            imputed,
            time_col=data.time_col,
            outcome_col=data.outcome_col,
            expl_vars=[],
            event_definition=data.event_definition,
        )

    cox_context = MethodContext(
        method="get_unique_event_times",
        meta={"df": imputed, "client": client},
    )
    cox_input = CoxGetUniqueEventTimesInput(
        time_col=data.time_col,
        outcome_col=data.outcome_col,
        minimum_events=data.minimum_events,
    )
    return cox_get_unique_event_times_handler(cox_input, context=cox_context)


def cox_compute_summed_z_imputed_handler(
    data: CoxComputeSummedZImputedInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for cox_compute_summed_z_imputed")

    df = _get_dataframe(context)
    strategy = _coerce_strategy(data.imputation_strategy)
    imputed = _filter_and_impute_for_survival(
        df,
        global_metrics=data.global_metrics,
        strategy=strategy,
        cohort=data.cohort,
    )
    if data.preprocess_raw_data:
        imputed = strata_fit_data_to_cox_input(
            imputed,
            outcome_col=data.outcome_col,
            expl_vars=data.expl_vars,
            event_definition=data.event_definition,
        )

    cox_context = MethodContext(
        method="compute_summed_z",
        meta={"df": imputed},
    )
    cox_input = CoxComputeSummedZInput(
        outcome_col=data.outcome_col,
        expl_vars=data.expl_vars,
    )
    return cox_compute_summed_z_handler(cox_input, context=cox_context)


def cox_perform_iteration_imputed_handler(
    data: CoxPerformIterationImputedInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for cox_perform_iteration_imputed")

    df = _get_dataframe(context)
    strategy = _coerce_strategy(data.imputation_strategy)
    imputed = _filter_and_impute_for_survival(
        df,
        global_metrics=data.global_metrics,
        strategy=strategy,
        cohort=data.cohort,
    )
    if data.preprocess_raw_data:
        imputed = strata_fit_data_to_cox_input(
            imputed,
            time_col=data.time_col,
            expl_vars=data.expl_vars,
            event_definition=data.event_definition,
        )

    cox_context = MethodContext(
        method="perform_iteration",
        meta={"df": imputed},
    )
    cox_input = CoxPerformIterationInput(
        time_col=data.time_col,
        expl_vars=data.expl_vars,
        beta=data.beta,
        unique_time_events=data.unique_time_events,
    )
    return cox_perform_iteration_handler(cox_input, context=cox_context)


def km_get_unique_event_times_imputed_handler(
    data: KMGetUniqueEventTimesImputedInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for km_get_unique_event_times_imputed")

    df = _get_dataframe(context)
    strategy = _coerce_strategy(data.imputation_strategy)
    imputed = _filter_and_impute_for_survival(
        df,
        global_metrics=data.global_metrics,
        strategy=strategy,
        cohort=data.cohort,
    )
    km_input = (
        strata_fit_data_to_km_input(imputed, event_definition=data.event_definition)
        if data.preprocess_raw_data
        else imputed
    )

    if km_input.empty:
        return {"times": []}

    start_times = km_input[DEFAULT_INTERVAL_START_COLUMN].dropna().tolist()
    end_times = km_input[DEFAULT_INTERVAL_END_COLUMN].dropna().tolist()
    unique_times = sorted({float(value) for value in [*start_times, *end_times]})
    return {"times": unique_times}


def _build_km_event_table(km_df: pd.DataFrame, unique_event_times: List[float]) -> pd.DataFrame:
    if not unique_event_times:
        return pd.DataFrame(
            columns=[
                DEFAULT_INTERVAL_START_COLUMN,
                "removed",
                "observed",
                "interval",
                "censored",
                "at_risk",
            ]
        )

    event_table = (
        pd.DataFrame({DEFAULT_INTERVAL_START_COLUMN: sorted(unique_event_times)})
        .drop_duplicates(subset=DEFAULT_INTERVAL_START_COLUMN)
        .reset_index(drop=True)
    )

    exact_events = km_df[km_df[DEFAULT_EVENT_INDICATOR_COLUMN] == "exact"]
    censored_events = km_df[km_df[DEFAULT_EVENT_INDICATOR_COLUMN] == "censored"]
    interval_events = km_df[km_df[DEFAULT_EVENT_INDICATOR_COLUMN] == "interval"]

    event_counts = (
        exact_events[DEFAULT_INTERVAL_START_COLUMN]
        .value_counts()
        .reindex(event_table[DEFAULT_INTERVAL_START_COLUMN], fill_value=0)
    )
    censored_counts = (
        censored_events[DEFAULT_INTERVAL_START_COLUMN]
        .value_counts()
        .reindex(event_table[DEFAULT_INTERVAL_START_COLUMN], fill_value=0)
    )
    interval_counts = (
        interval_events[DEFAULT_INTERVAL_END_COLUMN]
        .value_counts()
        .reindex(event_table[DEFAULT_INTERVAL_START_COLUMN], fill_value=0)
    )

    removed = event_counts + censored_counts + interval_counts
    event_table["removed"] = removed.to_numpy(dtype=float)
    event_table["observed"] = event_counts.to_numpy(dtype=float)
    event_table["interval"] = interval_counts.to_numpy(dtype=float)
    event_table["censored"] = censored_counts.to_numpy(dtype=float)
    event_table["at_risk"] = event_table["removed"].iloc[::-1].cumsum().iloc[::-1]
    return event_table


def km_get_event_table_imputed_handler(
    data: KMGetEventTableImputedInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for km_get_event_table_imputed")

    df = _get_dataframe(context)
    strategy = _coerce_strategy(data.imputation_strategy)
    imputed = _filter_and_impute_for_survival(
        df,
        global_metrics=data.global_metrics,
        strategy=strategy,
        cohort=data.cohort,
    )
    km_input = (
        strata_fit_data_to_km_input(imputed, event_definition=data.event_definition)
        if data.preprocess_raw_data
        else imputed
    )

    table = _build_km_event_table(km_input, data.unique_event_times)
    return {"table": table.to_dict(orient="list")}


def prevalence_by_year_imputed_handler(
    data: PrevalenceByYearImputedInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for prevalence_by_year_imputed")

    df = _get_dataframe(context)
    strategy = _coerce_strategy(data.imputation_strategy)
    imputed = _filter_and_impute_for_survival(
        df,
        global_metrics=data.global_metrics,
        strategy=strategy,
        cohort=data.cohort,
    )
    prevalence = compute_d2t_prevalence_by_year(
        imputed,
        event_definition=data.event_definition,
    )
    return {"rows": prevalence.to_dict(orient="records")}


def _cox_input_for_risk_summary(
    df: pd.DataFrame,
    *,
    global_metrics: Dict[str, Any],
    strategy: ImputationStrategyEnum,
    cohort: Dict[str, Any],
    time_col: str,
    outcome_col: str,
    expl_vars: List[str],
    preprocess_raw_data: bool,
    event_definition: str,
) -> pd.DataFrame:
    imputed = _filter_and_impute_for_survival(
        df,
        global_metrics=global_metrics,
        strategy=strategy,
        cohort=cohort,
    )
    if preprocess_raw_data:
        return strata_fit_data_to_cox_input(
            imputed,
            time_col=time_col,
            outcome_col=outcome_col,
            expl_vars=expl_vars,
            event_definition=event_definition,
        )
    return imputed


def cox_risk_score_range_imputed_handler(
    data: CoxRiskScoreRangeImputedInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for cox_risk_score_range_imputed")

    df = _get_dataframe(context)
    strategy = _coerce_strategy(data.imputation_strategy)
    cox_input = _cox_input_for_risk_summary(
        df,
        global_metrics=data.global_metrics,
        strategy=strategy,
        cohort=data.cohort,
        time_col=data.time_col,
        outcome_col=data.outcome_col,
        expl_vars=data.expl_vars,
        preprocess_raw_data=data.preprocess_raw_data,
        event_definition=data.event_definition,
    )
    if cox_input.empty:
        return {"min_score": None, "max_score": None, "count": 0}
    working = cox_input.dropna(subset=data.expl_vars, how="any")
    if working.empty:
        return {"min_score": None, "max_score": None, "count": 0}
    beta = np.asarray(data.beta, dtype=float)
    scores = working[data.expl_vars].to_numpy(dtype=float) @ beta
    return {
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "count": int(scores.shape[0]),
    }


def cox_risk_score_histogram_imputed_handler(
    data: CoxRiskScoreHistogramImputedInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for cox_risk_score_histogram_imputed")

    df = _get_dataframe(context)
    strategy = _coerce_strategy(data.imputation_strategy)
    cox_input = _cox_input_for_risk_summary(
        df,
        global_metrics=data.global_metrics,
        strategy=strategy,
        cohort=data.cohort,
        time_col=data.time_col,
        outcome_col=data.outcome_col,
        expl_vars=data.expl_vars,
        preprocess_raw_data=data.preprocess_raw_data,
        event_definition=data.event_definition,
    )
    if cox_input.empty:
        return {"counts": [0 for _ in range(max(len(data.bin_edges) - 1, 0))]}
    working = cox_input.dropna(subset=data.expl_vars, how="any")
    if working.empty:
        return {"counts": [0 for _ in range(max(len(data.bin_edges) - 1, 0))]}
    beta = np.asarray(data.beta, dtype=float)
    scores = working[data.expl_vars].to_numpy(dtype=float) @ beta
    counts, _ = np.histogram(scores, bins=np.asarray(data.bin_edges, dtype=float))
    return {"counts": counts.astype(int).tolist()}


def cox_risk_group_summary_imputed_handler(
    data: CoxRiskGroupSummaryImputedInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for cox_risk_group_summary_imputed")

    df = _get_dataframe(context)
    strategy = _coerce_strategy(data.imputation_strategy)
    cox_input = _cox_input_for_risk_summary(
        df,
        global_metrics=data.global_metrics,
        strategy=strategy,
        cohort=data.cohort,
        time_col=data.time_col,
        outcome_col=data.outcome_col,
        expl_vars=data.expl_vars,
        preprocess_raw_data=data.preprocess_raw_data,
        event_definition=data.event_definition,
    )
    if cox_input.empty:
        return {"groups": []}
    working = cox_input.dropna(subset=[data.time_col, data.outcome_col, *data.expl_vars], how="any").copy()
    if working.empty:
        return {"groups": []}

    beta = np.asarray(data.beta, dtype=float)
    scores = working[data.expl_vars].to_numpy(dtype=float) @ beta
    cutoffs = sorted(float(cutoff) for cutoff in data.cutoffs[:2])
    lower_cutoff = cutoffs[0] if cutoffs else float("-inf")
    upper_cutoff = cutoffs[1] if len(cutoffs) > 1 else float("inf")

    def assign_group(score: float) -> str:
        if score <= lower_cutoff:
            return "low"
        if score <= upper_cutoff:
            return "intermediate"
        return "high"

    working["risk_group"] = [assign_group(float(score)) for score in scores]
    horizons = sorted({int(horizon) for horizon in data.horizons_months if int(horizon) > 0})

    groups: list[dict[str, Any]] = []
    for label in ("low", "intermediate", "high"):
        group = working[working["risk_group"] == label]
        if group.empty:
            groups.append(
                {
                    "label": label,
                    "count": 0,
                    "events": 0,
                    "event_time_counts": {},
                    "censor_time_counts": {},
                }
            )
            continue
        events = int(group[data.outcome_col].sum())
        event_time_counts = (
            group[group[data.outcome_col] == 1]
            .groupby(data.time_col)
            .size()
            .to_dict()
        )
        censor_time_counts = (
            group[group[data.outcome_col] != 1]
            .groupby(data.time_col)
            .size()
            .to_dict()
        )
        groups.append(
            {
                "label": label,
                "count": int(group.shape[0]),
                "events": events,
                "horizons_months": horizons,
                "event_time_counts": {str(float(key)): int(value) for key, value in event_time_counts.items()},
                "censor_time_counts": {str(float(key)): int(value) for key, value in censor_time_counts.items()},
            }
        )
    return {"groups": groups}


def _aggregate_sklearn_linear_models(partials: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not partials:
        return {}

    total = sum(max(int(partial.get("size", 0)), 0) for partial in partials)
    first = partials[0].get("model_attributes", {})
    if total <= 0:
        return first

    required_keys = ("coef_", "intercept_")
    for key in required_keys:
        if key not in first:
            raise DataContractError(f"Missing key '{key}' in sklearn-linear partial result")

    coef_sum = np.zeros_like(np.asarray(first["coef_"], dtype=float))
    inter_sum = np.zeros_like(np.asarray(first["intercept_"], dtype=float))

    for partial in partials:
        weight = max(int(partial.get("size", 0)), 0)
        attrs = partial.get("model_attributes", {})
        coef = np.asarray(attrs.get("coef_"), dtype=float)
        intercept = np.asarray(attrs.get("intercept_"), dtype=float)
        if coef.shape != coef_sum.shape or intercept.shape != inter_sum.shape:
            raise DataContractError("Inconsistent sklearn-linear model shapes across nodes")
        coef_sum += coef * weight
        inter_sum += intercept * weight

    aggregated = {
        "coef_": (coef_sum / total).tolist(),
        "intercept_": (inter_sum / total).tolist(),
    }
    if "classes_" in first:
        aggregated["classes_"] = first["classes_"]
    return aggregated


def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        warn("Matrix inversion failed; using pseudo-inverse")
        return np.linalg.pinv(matrix)


def _compute_log_likelihood(
    z_sum: pd.Series,
    beta: np.ndarray,
    summed_agg1: np.ndarray,
    aggregated_time_events: pd.DataFrame,
) -> float:
    linear_part = float(np.dot(z_sum.to_numpy(dtype=float), beta))
    risk_set_part = 0.0
    for i, row in aggregated_time_events.iterrows():
        if i >= len(summed_agg1):
            break
        denom = float(summed_agg1[i])
        if denom <= 0.0:
            continue
        risk_set_part += float(row["freq"]) * math.log(denom)
    return linear_part - risk_set_part


def _run_cox_with_imputation(
    client: Any,
    org_ids: List[int],
    config: CoxFinalConfig,
    global_metrics: Dict[str, Any],
    strategy: ImputationStrategyEnum,
) -> Dict[str, Any]:
    runner = TaskRunner(client)
    ids = list(org_ids)
    excluded_ids: List[int] = []
    use_preprocess = bool(config.preprocess_raw_data)
    retried_without_preprocess = False

    unique_step = WorkflowStepSpec(
        name="cox-unique-events",
        method="cox_get_unique_event_times_imputed",
        input_model=CoxGetUniqueEventTimesImputedInput,
        output_model=CoxGetUniqueEventTimesImputedOutput,
    )
    summed_z_step = WorkflowStepSpec(
        name="cox-summed-z",
        method="cox_compute_summed_z_imputed",
        input_model=CoxComputeSummedZImputedInput,
        output_model=CoxComputeSummedZImputedOutput,
    )
    iter_step = WorkflowStepSpec(
        name="cox-iteration",
        method="cox_perform_iteration_imputed",
        input_model=CoxPerformIterationImputedInput,
        output_model=CoxPerformIterationImputedOutput,
    )

    n_covs = len(config.expl_vars)
    max_iterations = max(1, int(config.max_iterations))
    tolerance = float(config.tolerance)

    unique_time_events: List[float] = []
    aggregated_time_events = pd.DataFrame(columns=[config.time_col, "freq"])

    n_loops = 0
    while True:
        if n_loops >= MAX_N_THRESHOLD_RETRIES:
            raise DataContractError("Sample size threshold could not be met after retries")
        n_loops += 1

        results = runner.run(
            unique_step,
            {
                "time_col": config.time_col,
                "outcome_col": config.outcome_col,
                "minimum_events": 10,
                "global_metrics": global_metrics,
                "imputation_strategy": strategy,
                "preprocess_raw_data": use_preprocess,
                "cohort": config.cohort,
                "event_definition": config.event_definition,
            },
            ids,
        )

        unique_frames: List[pd.DataFrame] = []
        loop_excluded: List[int] = []
        for output in results:
            not_met = output.get("n_threshold_not_met")
            if not_met is not None:
                if not_met in ids:
                    ids.remove(not_met)
                excluded_ids.append(not_met)
                loop_excluded.append(not_met)
                continue

            times = output.get("times")
            if times:
                unique_frames.append(pd.DataFrame.from_dict(times))

        if loop_excluded and not ids:
            return {
                "included_organizations": [],
                "excluded_organizations": excluded_ids,
                "table": float("nan"),
                "coefficients": [],
                "warnings": ["No organizations met the minimum event threshold"],
            }

        if not loop_excluded:
            if unique_frames:
                aggregated_time_events = pd.concat(unique_frames)
                aggregated_time_events = (
                    aggregated_time_events.groupby(config.time_col, as_index=False).sum()
                )
                unique_time_events = aggregated_time_events[config.time_col].tolist()
                break

            # Some upstream preprocessing combinations can yield empty Cox event tables
            # even when raw event-time columns are present. Retry once on raw columns.
            if use_preprocess and not retried_without_preprocess:
                retried_without_preprocess = True
                use_preprocess = False
                continue
            break

    if not unique_time_events:
        if config.preprocess_raw_data:
            placeholder = pd.DataFrame(
                {
                    "Coef": np.zeros(len(config.expl_vars), dtype=float),
                    "Exp(coef)": np.ones(len(config.expl_vars), dtype=float),
                    "SE": np.zeros(len(config.expl_vars), dtype=float),
                    "Var": config.expl_vars,
                    "Z": np.zeros(len(config.expl_vars), dtype=float),
                    "p-value": np.ones(len(config.expl_vars), dtype=float),
                }
            ).set_index("Var")
            return {
                "included_organizations": ids,
                "excluded_organizations": excluded_ids,
                "model": placeholder.to_json(),
                "table": float("nan"),
                "coefficients": [
                    {
                        "variable": variable,
                        "coef": 0.0,
                        "hazard_ratio": 1.0,
                        "standard_error": 0.0,
                        "z_value": 0.0,
                        "p_value": 1.0,
                        "lower_ci": 1.0,
                        "upper_ci": 1.0,
                    }
                    for variable in config.expl_vars
                ],
                "warnings": [
                    "No event table rows were produced by Cox preprocessing; returned placeholder model"
                ],
            }
        return {
            "included_organizations": [],
            "excluded_organizations": excluded_ids,
            "table": float("nan"),
            "coefficients": [],
            "warnings": ["No organizations met the minimum event threshold"],
        }

    z_results = runner.run(
        summed_z_step,
        {
            "outcome_col": config.outcome_col,
            "expl_vars": config.expl_vars,
            "global_metrics": global_metrics,
            "imputation_strategy": strategy,
            "preprocess_raw_data": use_preprocess,
            "cohort": config.cohort,
            "event_definition": config.event_definition,
        },
        ids,
    )

    z_sum = pd.Series(0.0, index=config.expl_vars)
    for output in z_results:
        z_sum += pd.Series(output["sum"], index=config.expl_vars, dtype=float).fillna(0.0)

    beta = np.zeros(n_covs, dtype=float)
    secondary_derivative = -np.eye(n_covs, dtype=float)
    summed_agg1 = np.zeros(len(unique_time_events), dtype=float)

    for _ in range(max_iterations):
        results = runner.run(
            iter_step,
            {
                "time_col": config.time_col,
                "expl_vars": config.expl_vars,
                "beta": beta.tolist(),
                "unique_time_events": unique_time_events,
                "global_metrics": global_metrics,
                "imputation_strategy": strategy,
                "preprocess_raw_data": use_preprocess,
                "cohort": config.cohort,
                "event_definition": config.event_definition,
            },
            ids,
        )

        n_times = len(unique_time_events)
        summed_agg1 = np.zeros(n_times, dtype=float)
        summed_agg2 = np.zeros((n_times, n_covs), dtype=float)
        summed_agg3 = np.zeros((n_times, n_covs, n_covs), dtype=float)

        for output in results:
            summed_agg1 += np.asarray(output["agg1"], dtype=float)

            agg2_df = pd.DataFrame.from_dict(output["agg2"])
            agg2_df = agg2_df.reindex(columns=config.expl_vars)
            summed_agg2 += agg2_df.to_numpy(dtype=float)

            summed_agg3 += np.asarray(output["agg3"], dtype=float)

        primary_derivative, secondary_derivative = compute_derivatives(
            summed_agg1=summed_agg1,
            summed_agg2=summed_agg2,
            summed_agg3=summed_agg3,
            aggregated_time_events=aggregated_time_events,
            z_sum=z_sum,
        )

        beta_old = beta.copy()
        try:
            beta = beta_old - solve(secondary_derivative, primary_derivative)
        except np.linalg.LinAlgError:
            beta = beta_old - _safe_inverse(secondary_derivative).dot(primary_derivative)

        delta = float(np.max(np.abs(beta - beta_old)))
        if math.isnan(delta):
            warn("Optimization update produced NaN delta; stopping iterations")
            break
        if delta <= tolerance:
            info("Betas have settled; optimization converged")
            break

    fisher = _safe_inverse(-secondary_derivative)
    s_errors = np.sqrt(np.clip(np.diag(fisher), a_min=0.0, a_max=None))
    with np.errstate(divide="ignore", invalid="ignore"):
        zvalues = np.divide(
            beta,
            s_errors,
            out=np.zeros_like(beta, dtype=float),
            where=s_errors > 0,
        )
    pvalues = 2.0 * norm.sf(np.abs(zvalues))

    table = pd.DataFrame(
        {
            "Coef": np.around(beta, 5),
            "Exp(coef)": np.around(np.exp(beta), 5),
            "SE": np.around(s_errors, 5),
            "Var": config.expl_vars,
            "Z": zvalues,
            "p-value": pvalues,
        }
    ).set_index("Var")
    table["lower_CI"] = np.around(np.exp(table["Coef"] - 1.96 * table["SE"]), 5)
    table["upper_CI"] = np.around(np.exp(table["Coef"] + 1.96 * table["SE"]), 5)

    information = -secondary_derivative
    wald_statistic = float(beta @ information @ beta)
    overall_p_value = float(chi2.sf(wald_statistic, len(beta)))

    try:
        log_likelihood = _compute_log_likelihood(
            z_sum=z_sum,
            beta=beta,
            summed_agg1=summed_agg1,
            aggregated_time_events=aggregated_time_events,
        )
        if math.isnan(log_likelihood) or math.isinf(log_likelihood):
            raise ValueError(f"Invalid log-likelihood: {log_likelihood}")
        aic = float(-2.0 * log_likelihood + 2.0 * len(beta))
    except (ValueError, FloatingPointError):
        aic = float("nan")

    warnings_: List[str] = []
    for covariate, row in table.iterrows():
        coef = float(row["Coef"])
        se = float(row["SE"])
        if (
            abs(coef) > LARGE_VALUE_WARNING_THRESHOLD
            or np.isinf(coef)
            or np.isnan(coef)
            or abs(se) > LARGE_VALUE_WARNING_THRESHOLD
            or np.isinf(se)
            or np.isnan(se)
        ):
            warnings_.append(
                f"Covariate '{covariate}' may perfectly predict the event; results may be unreliable."
            )

    return {
        "included_organizations": ids,
        "excluded_organizations": excluded_ids,
        "model": table.to_json(),
        "coefficients": [
            {
                "variable": variable,
                "coef": float(row["Coef"]),
                "hazard_ratio": float(row["Exp(coef)"]),
                "standard_error": float(row["SE"]),
                "z_value": float(row["Z"]),
                "p_value": float(row["p-value"]),
                "lower_ci": float(row["lower_CI"]),
                "upper_ci": float(row["upper_CI"]),
            }
            for variable, row in table.iterrows()
        ],
        "overall_p_value": overall_p_value,
        "aic": aic,
        "degrees_of_freedom": int(len(beta)),
        "converged": bool("delta" in locals() and delta <= tolerance),
        "warnings": warnings_,
    }


def _run_km_with_imputation(
    client: Any,
    org_ids: List[int],
    config: KMFinalConfig,
    global_metrics: Dict[str, Any],
    strategy: ImputationStrategyEnum,
) -> Dict[str, Any]:
    runner = TaskRunner(client)

    unique_step = WorkflowStepSpec(
        name="km-unique-events",
        method="km_get_unique_event_times_imputed",
        input_model=KMGetUniqueEventTimesImputedInput,
        output_model=KMGetUniqueEventTimesImputedOutput,
    )
    table_step = WorkflowStepSpec(
        name="km-event-table",
        method="km_get_event_table_imputed",
        input_model=KMGetEventTableImputedInput,
        output_model=KMGetEventTableImputedOutput,
    )

    unique_results = runner.run(
        unique_step,
        {
            "global_metrics": global_metrics,
            "imputation_strategy": strategy,
            "preprocess_raw_data": config.preprocess_raw_data,
            "cohort": config.cohort,
            "event_definition": config.event_definition,
        },
        org_ids,
    )
    unique_event_times = sorted(
        {float(time_value) for output in unique_results for time_value in output.get("times", [])}
    )

    table_results = runner.run(
        table_step,
        {
            "unique_event_times": unique_event_times,
            "global_metrics": global_metrics,
            "imputation_strategy": strategy,
            "preprocess_raw_data": config.preprocess_raw_data,
            "cohort": config.cohort,
            "event_definition": config.event_definition,
        },
        org_ids,
    )

    tables = [pd.DataFrame(output.get("table", {})) for output in table_results]
    if tables:
        km_df = pd.concat(tables, ignore_index=True)
        km_df = (
            km_df.groupby(DEFAULT_INTERVAL_START_COLUMN, as_index=False)[
                ["removed", "observed", "interval", "censored", "at_risk"]
            ].sum()
        )
    else:
        km_df = pd.DataFrame(
            columns=[
                DEFAULT_INTERVAL_START_COLUMN,
                "removed",
                "observed",
                "interval",
                "censored",
                "at_risk",
            ]
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        km_df["hazard"] = np.divide(
            km_df["observed"] + 0.5 * km_df["interval"],
            km_df["at_risk"],
            out=np.zeros_like(km_df["at_risk"], dtype=float),
            where=km_df["at_risk"].to_numpy(dtype=float) > 0,
        )
    km_df["cumulative_incidence"] = 1 - (1 - km_df["hazard"]).cumprod()

    series = [
        {
            "time_months": float(row[DEFAULT_INTERVAL_START_COLUMN]),
            "cumulative_incidence": float(row["cumulative_incidence"]),
            "at_risk": int(row["at_risk"]),
            "observed": int(row["observed"]),
            "censored": int(row["censored"]),
            "interval": int(row["interval"]),
        }
        for _, row in km_df.iterrows()
    ]
    five_year_points = [point for point in series if point["time_months"] <= 60.0]
    five_year_cumulative_incidence = (
        float(five_year_points[-1]["cumulative_incidence"])
        if five_year_points
        else 0.0
    )

    return {
        "included_organizations": org_ids,
        "unique_event_times": unique_event_times,
        "km_curve": km_df.to_json(),
        "series": series,
        "five_year_cumulative_incidence": five_year_cumulative_incidence,
        "event_definition": config.event_definition,
    }


def _run_prevalence_with_imputation(
    client: Any,
    org_ids: List[int],
    *,
    cohort: Dict[str, Any],
    event_definition: str,
    global_metrics: Dict[str, Any],
    strategy: ImputationStrategyEnum,
) -> Dict[str, Any]:
    runner = TaskRunner(client)
    prevalence_step = WorkflowStepSpec(
        name="prevalence-by-year",
        method="prevalence_by_year_imputed",
        input_model=PrevalenceByYearImputedInput,
        output_model=PrevalenceByYearImputedOutput,
    )
    results = runner.run(
        prevalence_step,
        {
            "global_metrics": global_metrics,
            "imputation_strategy": strategy,
            "cohort": cohort,
            "event_definition": event_definition,
        },
        org_ids,
    )

    rows: list[dict[str, Any]] = []
    for output in results:
        rows.extend(output.get("rows", []))
    if not rows:
        return {
            "series": [],
            "latest_year": None,
            "latest_complete_year": None,
            "latest_complete_year_prevalence": None,
            "event_definition": event_definition,
        }

    prevalence_df = pd.DataFrame(rows)
    prevalence_df = (
        prevalence_df.groupby("Year_visit", as_index=False)
        .agg(
            total_patients=("total_patients", "sum"),
            d2t_positive=("d2t_positive", "sum"),
            partial_year=("partial_year", "max"),
        )
        .sort_values("Year_visit")
        .reset_index(drop=True)
    )
    prevalence_df["prevalence"] = np.divide(
        prevalence_df["d2t_positive"],
        prevalence_df["total_patients"],
        out=np.zeros_like(prevalence_df["d2t_positive"], dtype=float),
        where=prevalence_df["total_patients"].to_numpy(dtype=float) > 0,
    )

    series = [
        {
            "year": int(row["Year_visit"]),
            "total_patients": int(row["total_patients"]),
            "d2t_positive": int(row["d2t_positive"]),
            "prevalence": float(row["prevalence"]),
            "partial_year": bool(row["partial_year"]),
        }
        for _, row in prevalence_df.iterrows()
    ]
    complete_years = [row for row in series if not row["partial_year"]]
    latest_complete = complete_years[-1] if complete_years else None
    return {
        "series": series,
        "latest_year": int(series[-1]["year"]),
        "latest_complete_year": latest_complete["year"] if latest_complete else None,
        "latest_complete_year_prevalence": latest_complete["prevalence"] if latest_complete else None,
        "event_definition": event_definition,
    }


def _approximate_tertile_cutoffs(bin_edges: np.ndarray, counts: np.ndarray) -> list[float]:
    total = int(counts.sum())
    if total <= 0:
        return [0.0, 0.0]

    thresholds = [total / 3.0, (2.0 * total) / 3.0]
    cutoffs: list[float] = []
    cumulative = np.cumsum(counts.astype(float))
    for threshold in thresholds:
        index = int(np.searchsorted(cumulative, threshold, side="left"))
        edge_index = min(index + 1, len(bin_edges) - 1)
        cutoffs.append(float(bin_edges[edge_index]))
    while len(cutoffs) < 2:
        cutoffs.append(float(bin_edges[-1]))
    return cutoffs[:2]


def _km_cumulative_incidence_at_horizon(
    event_time_counts: dict[float, int],
    censor_time_counts: dict[float, int],
    count: int,
    horizon: int,
) -> tuple[int, float]:
    if count <= 0:
        return 0, 0.0

    survival = 1.0
    at_risk = float(count)
    cumulative_events = 0
    all_times = sorted({*event_time_counts.keys(), *censor_time_counts.keys()})
    for time_point in all_times:
        if float(time_point) > float(horizon):
            break
        observed = int(event_time_counts.get(time_point, 0))
        censored = int(censor_time_counts.get(time_point, 0))
        if observed > 0 and at_risk > 0:
            survival *= max(0.0, 1.0 - (observed / at_risk))
            cumulative_events += observed
        at_risk -= observed + censored
        if at_risk <= 0:
            at_risk = 0.0
            break
    return cumulative_events, float(1.0 - survival)


def _run_risk_stratification_with_imputation(
    client: Any,
    org_ids: List[int],
    *,
    config: SurvivalBundleFinalConfig,
    beta: List[float],
    global_metrics: Dict[str, Any],
    strategy: ImputationStrategyEnum,
) -> Dict[str, Any]:
    if not beta:
        return {"cutoffs": [], "groups": []}

    runner = TaskRunner(client)
    range_step = WorkflowStepSpec(
        name="cox-risk-range",
        method="cox_risk_score_range_imputed",
        input_model=CoxRiskScoreRangeImputedInput,
        output_model=CoxRiskScoreRangeImputedOutput,
    )
    histogram_step = WorkflowStepSpec(
        name="cox-risk-histogram",
        method="cox_risk_score_histogram_imputed",
        input_model=CoxRiskScoreHistogramImputedInput,
        output_model=CoxRiskScoreHistogramImputedOutput,
    )
    group_step = WorkflowStepSpec(
        name="cox-risk-groups",
        method="cox_risk_group_summary_imputed",
        input_model=CoxRiskGroupSummaryImputedInput,
        output_model=CoxRiskGroupSummaryImputedOutput,
    )

    range_results = runner.run(
        range_step,
        {
            "time_col": config.time_col,
            "outcome_col": config.outcome_col,
            "expl_vars": config.expl_vars,
            "beta": beta,
            "global_metrics": global_metrics,
            "imputation_strategy": strategy,
            "preprocess_raw_data": config.preprocess_raw_data,
            "cohort": config.cohort,
            "event_definition": config.event_definition,
        },
        org_ids,
    )
    mins = [output["min_score"] for output in range_results if output.get("min_score") is not None]
    maxes = [output["max_score"] for output in range_results if output.get("max_score") is not None]
    if not mins or not maxes:
        return {"cutoffs": [], "groups": []}

    min_score = float(min(mins))
    max_score = float(max(maxes))
    if min_score == max_score:
        cutoffs = [min_score, max_score]
    else:
        bin_edges = np.linspace(min_score, max_score, num=61)
        histogram_results = runner.run(
            histogram_step,
            {
                "time_col": config.time_col,
                "outcome_col": config.outcome_col,
                "expl_vars": config.expl_vars,
                "beta": beta,
                "bin_edges": bin_edges.tolist(),
                "global_metrics": global_metrics,
                "imputation_strategy": strategy,
                "preprocess_raw_data": config.preprocess_raw_data,
                "cohort": config.cohort,
                "event_definition": config.event_definition,
            },
            org_ids,
        )
        histogram_arrays = [np.asarray(output.get("counts", []), dtype=int) for output in histogram_results]
        if not histogram_arrays:
            return {"cutoffs": [], "groups": []}
        counts = np.sum(histogram_arrays, axis=0)
        cutoffs = _approximate_tertile_cutoffs(bin_edges, counts)

    group_results = runner.run(
        group_step,
        {
            "time_col": config.time_col,
            "outcome_col": config.outcome_col,
            "expl_vars": config.expl_vars,
            "beta": beta,
            "cutoffs": cutoffs,
            "horizons_months": config.horizons_months,
            "global_metrics": global_metrics,
            "imputation_strategy": strategy,
            "preprocess_raw_data": config.preprocess_raw_data,
            "cohort": config.cohort,
            "event_definition": config.event_definition,
        },
        org_ids,
    )

    combined: dict[str, dict[str, Any]] = {}
    for output in group_results:
        for group in output.get("groups", []):
            label = str(group["label"])
            entry = combined.setdefault(
                label,
                {
                    "label": label,
                    "count": 0,
                    "events": 0,
                    "event_time_counts": {},
                    "censor_time_counts": {},
                },
            )
            entry["count"] += int(group.get("count", 0))
            entry["events"] += int(group.get("events", 0))
            for time_point, value in (group.get("event_time_counts") or {}).items():
                normalized_time = float(time_point)
                entry["event_time_counts"][normalized_time] = entry["event_time_counts"].get(normalized_time, 0) + int(value)
            for time_point, value in (group.get("censor_time_counts") or {}).items():
                normalized_time = float(time_point)
                entry["censor_time_counts"][normalized_time] = entry["censor_time_counts"].get(normalized_time, 0) + int(value)

    groups = []
    horizons = sorted({int(horizon) for horizon in config.horizons_months if int(horizon) > 0})
    for label in ("low", "intermediate", "high"):
        entry = combined.get(
            label,
            {
                "label": label,
                "count": 0,
                "events": 0,
                "event_time_counts": {},
                "censor_time_counts": {},
            },
        )
        horizons = []
        for months in sorted({int(horizon) for horizon in config.horizons_months if int(horizon) > 0}):
            event_count, cumulative_incidence = _km_cumulative_incidence_at_horizon(
                entry["event_time_counts"],
                entry["censor_time_counts"],
                int(entry["count"]),
                months,
            )
            horizons.append(
                {
                    "months": months,
                    "event_count": event_count,
                    "count": int(entry["count"]),
                    "cumulative_incidence": cumulative_incidence,
                }
            )
        groups.append(
            {
                "label": label,
                "count": int(entry["count"]),
                "events": int(entry["events"]),
                "horizons": horizons,
            }
        )
    return {"cutoffs": cutoffs, "groups": groups}


def _run_definition_sensitivity(
    client: Any,
    org_ids: List[int],
    *,
    config: SurvivalBundleFinalConfig,
    global_metrics: Dict[str, Any],
    strategy: ImputationStrategyEnum,
) -> Dict[str, Any]:
    definitions = [item for item in list_supported_d2t_definitions() if item["id"] != config.event_definition]
    summary_rows: list[dict[str, Any]] = []
    for definition in definitions:
        definition_id = str(definition["id"])
        km_result = _run_km_with_imputation(
            client,
            org_ids,
            KMFinalConfig(
                preprocess_raw_data=config.preprocess_raw_data,
                cohort=config.cohort,
                event_definition=definition_id,
            ),
            global_metrics,
            strategy,
        )
        prevalence_result = _run_prevalence_with_imputation(
            client,
            org_ids,
            cohort=config.cohort,
            event_definition=definition_id,
            global_metrics=global_metrics,
            strategy=strategy,
        )
        latest_complete = prevalence_result.get("latest_complete_year_prevalence")
        series = prevalence_result.get("series", [])
        mean_prevalence = float(np.mean([row["prevalence"] for row in series])) if series else 0.0
        summary_rows.append(
            {
                "definition_id": definition_id,
                "label": definition["label"],
                "five_year_cumulative_incidence": km_result.get("five_year_cumulative_incidence", 0.0),
                "latest_complete_year_prevalence": float(latest_complete) if latest_complete is not None else None,
                "mean_annual_prevalence": mean_prevalence,
            }
        )
    return {"definitions": summary_rows}


def _run_survival_bundle_with_imputation(
    client: Any,
    org_ids: List[int],
    *,
    config: SurvivalBundleFinalConfig,
    global_metrics: Dict[str, Any],
    strategy: ImputationStrategyEnum,
) -> Dict[str, Any]:
    incidence = _run_km_with_imputation(
        client,
        org_ids,
        KMFinalConfig(
            preprocess_raw_data=config.preprocess_raw_data,
            cohort=config.cohort,
            event_definition=config.event_definition,
        ),
        global_metrics,
        strategy,
    )
    cox = _run_cox_with_imputation(
        client,
        org_ids,
        CoxFinalConfig(
            time_col=config.time_col,
            outcome_col=config.outcome_col,
            expl_vars=config.expl_vars,
            max_iterations=config.max_iterations,
            tolerance=config.tolerance,
            preprocess_raw_data=config.preprocess_raw_data,
            cohort=config.cohort,
            event_definition=config.event_definition,
        ),
        global_metrics,
        strategy,
    )
    prevalence = _run_prevalence_with_imputation(
        client,
        org_ids,
        cohort=config.cohort,
        event_definition=config.event_definition,
        global_metrics=global_metrics,
        strategy=strategy,
    )
    risk = _run_risk_stratification_with_imputation(
        client,
        cox.get("included_organizations", org_ids),
        config=config,
        beta=[row["coef"] for row in cox.get("coefficients", [])],
        global_metrics=global_metrics,
        strategy=strategy,
    )
    out = {
        "definition": {
            "id": config.event_definition,
            "label": get_d2t_definition_config(config.event_definition).label,
            "supported_definitions": list_supported_d2t_definitions(),
            "cohort": config.cohort,
            "horizons_months": config.horizons_months,
        },
        "incidence": incidence,
        "prevalence": prevalence,
        "cox": cox,
        "risk_stratification": risk,
        "metadata": {
            "event_definition": config.event_definition,
            "included_organizations": cox.get("included_organizations", org_ids),
            "excluded_organizations": cox.get("excluded_organizations", []),
            "cohort": config.cohort,
        },
    }
    if config.include_definition_sensitivity:
        out["definition_sensitivity"] = _run_definition_sensitivity(
            client,
            org_ids,
            config=config,
            global_metrics=global_metrics,
            strategy=strategy,
        )
    return out


def central_handler(
    data: MetaCentralInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for central")

    client = _get_client(context)
    org_ids = _resolve_organization_ids(client, data.organizations, context)
    strategy = _coerce_strategy(data.imputation_strategy)
    runner = TaskRunner(client)

    validate_step = WorkflowStepSpec(
        name="validate",
        method="validate_partial",
        input_model=ValidatePartialInput,
        output_model=ValidatePartialOutput,
    )
    impute_step = WorkflowStepSpec(
        name="imputation-compute",
        method="imputation_compute_partial",
        input_model=ImputationComputePartialInput,
        output_model=ImputationComputePartialOutput,
    )
    linear_step = WorkflowStepSpec(
        name="sklearn-linear-fit",
        method="impute_and_train_sklearn_linear",
        input_model=ImputeAndTrainSklearnLinearInput,
        output_model=ImputeAndTrainSklearnLinearOutput,
    )

    results: Dict[str, Any] = {"organizations": org_ids, "validation": []}
    if data.run_validation:
        results["validation"] = runner.run(
            validate_step,
            {"model_name": data.model_name},
            org_ids,
        )

    node_metrics = runner.run(
        impute_step,
        {
            "columns": data.columns,
            "imputation_strategy": strategy,
        },
        org_ids,
    )

    imputer_cls = STRATEGY_REGISTRY[strategy]
    imputer = imputer_cls()
    global_metrics = imputer.aggregate(node_metrics=node_metrics, columns=data.columns)
    results["imputation_metrics"] = global_metrics

    if data.final_model == FinalModelEnum.SKLEARN_LINEAR:
        linear_config = SklearnLinearFinalConfig.model_validate(data.final_model_config)
        partials = runner.run(
            linear_step,
            {
                "global_metrics": global_metrics,
                "predictors": linear_config.predictors,
                "outcome": linear_config.outcome,
                "n_local_iterations": linear_config.n_local_iterations,
                "model_class": linear_config.model_class,
                "model_kwargs": linear_config.model_kwargs,
                "imputation_strategy": strategy,
            },
            org_ids,
        )
        results["final_result"] = {
            "partials": partials,
            "model_attributes": _aggregate_sklearn_linear_models(partials),
        }
    elif data.final_model == FinalModelEnum.COX:
        cox_config = CoxFinalConfig.model_validate(data.final_model_config)
        results["final_result"] = _run_cox_with_imputation(
            client=client,
            org_ids=org_ids,
            config=cox_config,
            global_metrics=global_metrics,
            strategy=strategy,
        )
    elif data.final_model == FinalModelEnum.KM:
        km_config = KMFinalConfig.model_validate(data.final_model_config)
        results["final_result"] = _run_km_with_imputation(
            client=client,
            org_ids=org_ids,
            config=km_config,
            global_metrics=global_metrics,
            strategy=strategy,
        )
    elif data.final_model == FinalModelEnum.SURVIVAL_BUNDLE:
        survival_config = SurvivalBundleFinalConfig.model_validate(data.final_model_config)
        results["final_result"] = _run_survival_bundle_with_imputation(
            client=client,
            org_ids=org_ids,
            config=survival_config,
            global_metrics=global_metrics,
            strategy=strategy,
        )
        results["final_result"]["validation"] = results["validation"]
        results["final_result"]["imputation"] = global_metrics
    else:
        raise DataContractError(f"Unsupported final model: {data.final_model}")

    results["final_model"] = data.final_model
    return results


METHOD_REGISTRY = MethodRegistry(
    [
        MethodSpec(
            name="main",
            input_model=MetaCentralInput,
            output_model=MetaCentralOutput,
            handler=central_handler,
        ),
        MethodSpec(
            name="validate_partial",
            input_model=ValidatePartialInput,
            output_model=ValidatePartialOutput,
            handler=validate_partial_handler,
        ),
        MethodSpec(
            name="imputation_compute_partial",
            input_model=ImputationComputePartialInput,
            output_model=ImputationComputePartialOutput,
            handler=imputation_compute_partial_handler,
        ),
        MethodSpec(
            name="impute_and_train_sklearn_linear",
            input_model=ImputeAndTrainSklearnLinearInput,
            output_model=ImputeAndTrainSklearnLinearOutput,
            handler=impute_and_train_sklearn_linear_handler,
        ),
        MethodSpec(
            name="cox_get_unique_event_times_imputed",
            input_model=CoxGetUniqueEventTimesImputedInput,
            output_model=CoxGetUniqueEventTimesImputedOutput,
            handler=cox_get_unique_event_times_imputed_handler,
        ),
        MethodSpec(
            name="cox_compute_summed_z_imputed",
            input_model=CoxComputeSummedZImputedInput,
            output_model=CoxComputeSummedZImputedOutput,
            handler=cox_compute_summed_z_imputed_handler,
        ),
        MethodSpec(
            name="cox_perform_iteration_imputed",
            input_model=CoxPerformIterationImputedInput,
            output_model=CoxPerformIterationImputedOutput,
            handler=cox_perform_iteration_imputed_handler,
        ),
        MethodSpec(
            name="km_get_unique_event_times_imputed",
            input_model=KMGetUniqueEventTimesImputedInput,
            output_model=KMGetUniqueEventTimesImputedOutput,
            handler=km_get_unique_event_times_imputed_handler,
        ),
        MethodSpec(
            name="km_get_event_table_imputed",
            input_model=KMGetEventTableImputedInput,
            output_model=KMGetEventTableImputedOutput,
            handler=km_get_event_table_imputed_handler,
        ),
        MethodSpec(
            name="prevalence_by_year_imputed",
            input_model=PrevalenceByYearImputedInput,
            output_model=PrevalenceByYearImputedOutput,
            handler=prevalence_by_year_imputed_handler,
        ),
        MethodSpec(
            name="cox_risk_score_range_imputed",
            input_model=CoxRiskScoreRangeImputedInput,
            output_model=CoxRiskScoreRangeImputedOutput,
            handler=cox_risk_score_range_imputed_handler,
        ),
        MethodSpec(
            name="cox_risk_score_histogram_imputed",
            input_model=CoxRiskScoreHistogramImputedInput,
            output_model=CoxRiskScoreHistogramImputedOutput,
            handler=cox_risk_score_histogram_imputed_handler,
        ),
        MethodSpec(
            name="cox_risk_group_summary_imputed",
            input_model=CoxRiskGroupSummaryImputedInput,
            output_model=CoxRiskGroupSummaryImputedOutput,
            handler=cox_risk_group_summary_imputed_handler,
        ),
    ]
)


def build_policy_context(method_name: str, organization_ids: List[int]) -> PolicyContext:
    return PolicyContext(
        scope=PolicyScope.ELIGIBILITY,
        method=method_name,
        organization_count=len(organization_ids),
    )


def build_min_organization_policies() -> List[MinOrganizationsPolicy]:
    return [MinOrganizationsPolicy(minimum=MINIMUM_ORGANIZATIONS)]
