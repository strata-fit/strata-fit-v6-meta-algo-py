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
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info, warn

from coxph.contracts import (
    ComputeSummedZInput as CoxComputeSummedZInput,
    GetUniqueEventTimesInput as CoxGetUniqueEventTimesInput,
    PerformIterationInput as CoxPerformIterationInput,
)
from coxph.methods import (
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

from strata_fit_v6_data_validator_py.logic import load_data_models_from_settings, validate_csv
from strata_fit_v6_imputation_py.imputation_strategies.base import (
    ImputationStrategyEnum,
    STRATEGY_REGISTRY,
)
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
    SklearnLinearFinalConfig,
    ValidatePartialInput,
    ValidatePartialOutput,
)
from .preprocessing import (
    DEFAULT_EVENT_INDICATOR_COLUMN,
    DEFAULT_INTERVAL_END_COLUMN,
    DEFAULT_INTERVAL_START_COLUMN,
    strata_fit_data_to_cox_input,
    strata_fit_data_to_km_input,
)

MAX_N_THRESHOLD_RETRIES = 3
LARGE_VALUE_WARNING_THRESHOLD = 10.0
MINIMUM_ORGANIZATIONS = 1


def _get_client(context: MethodContext) -> AlgorithmClient:
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
    client: AlgorithmClient,
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
    return imputer.impute(df.copy(), global_metrics)


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
    working = df.dropna(how="any")
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
    return imputer.compute(df, data.columns).to_dict()


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
    imputed = _impute_locally(df, data.global_metrics, strategy)
    if data.preprocess_raw_data:
        imputed = strata_fit_data_to_cox_input(
            imputed,
            time_col=data.time_col,
            outcome_col=data.outcome_col,
            expl_vars=[],
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
    imputed = _impute_locally(df, data.global_metrics, strategy)
    if data.preprocess_raw_data:
        imputed = strata_fit_data_to_cox_input(
            imputed,
            outcome_col=data.outcome_col,
            expl_vars=data.expl_vars,
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
    imputed = _impute_locally(df, data.global_metrics, strategy)
    if data.preprocess_raw_data:
        imputed = strata_fit_data_to_cox_input(
            imputed,
            time_col=data.time_col,
            expl_vars=data.expl_vars,
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
    imputed = _impute_locally(df, data.global_metrics, strategy)
    km_input = strata_fit_data_to_km_input(imputed) if data.preprocess_raw_data else imputed

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
    imputed = _impute_locally(df, data.global_metrics, strategy)
    km_input = strata_fit_data_to_km_input(imputed) if data.preprocess_raw_data else imputed

    table = _build_km_event_table(km_input, data.unique_event_times)
    return {"table": table.to_dict(orient="list")}


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
    client: AlgorithmClient,
    org_ids: List[int],
    config: CoxFinalConfig,
    global_metrics: Dict[str, Any],
    strategy: ImputationStrategyEnum,
) -> Dict[str, Any]:
    runner = TaskRunner(client)
    ids = list(org_ids)
    excluded_ids: List[int] = []

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
                "preprocess_raw_data": config.preprocess_raw_data,
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

    z_results = runner.run(
        summed_z_step,
        {
            "outcome_col": config.outcome_col,
            "expl_vars": config.expl_vars,
            "global_metrics": global_metrics,
            "imputation_strategy": strategy,
            "preprocess_raw_data": config.preprocess_raw_data,
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
                "preprocess_raw_data": config.preprocess_raw_data,
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
        "overall_p_value": overall_p_value,
        "aic": aic,
        "degrees_of_freedom": int(len(beta)),
        "warnings": warnings_,
    }


def _run_km_with_imputation(
    client: AlgorithmClient,
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

    return {
        "included_organizations": org_ids,
        "unique_event_times": unique_event_times,
        "km_curve": km_df.to_json(),
    }


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
