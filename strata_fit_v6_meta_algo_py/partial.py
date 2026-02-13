"""
Meta-algorithm partials exposed to Vantage6 and helper functions for local tests.
"""
from typing import Any, Dict, List, Optional
import pandas as pd
from vantage6.algorithm.tools.decorators import data

from strata_fit_v6_imputation_py.imputation_strategies.base import (
    ImputationStrategyEnum,
    STRATEGY_REGISTRY,
)
from strata_fit_v6_data_validator_py.logic import (
    load_data_models_from_settings,
    validate_csv,
)
from v6_logistic_regression_py.partials import _logistic_regression_partial
from strata_fit_v6_km_py.preprocessing import strata_fit_data_to_km_input
from strata_fit_v6_km_py.types import DEFAULT_INTERVAL_START_COLUMN, DEFAULT_INTERVAL_END_COLUMN
from strata_fit_v6_km_py.types import DEFAULT_EVENT_INDICATOR_COLUMN


# ---------- Helper (undecorated) ----------
def impute_locally(
    df: pd.DataFrame,
    global_metrics: Dict[str, Any],
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER,
) -> pd.DataFrame:
    """Apply global metrics to a local dataframe using the chosen strategy."""
    imputer_cls = STRATEGY_REGISTRY[imputation_strategy]
    imputer = imputer_cls()
    return imputer.impute(df, global_metrics)


def _impute_and_train_lr_core(
    df: pd.DataFrame,
    global_metrics: Dict[str, Any],
    predictors: List[str],
    outcome: str,
    n_local_iterations: int = 50,
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER,
    run_km: bool = True,
    unique_event_times: Optional[List[float]] = None,
    km_noise_type=None,
    km_snr: Optional[float] = None,
    km_random_seed: Optional[int] = None,
    **model_kwargs,
) -> Dict[str, Any]:
    # 1) Impute
    imputed = impute_locally(
        df,
        global_metrics,
        imputation_strategy=imputation_strategy,
    )
    km_summary = strata_fit_data_to_km_input(imputed)
    # 2) Optional KM
      # 2) Optional KM
    km_payload: Dict[str, Any] = {}
    if run_km:
        if unique_event_times is None:
            unique_times = pd.concat([
                km_summary[DEFAULT_INTERVAL_START_COLUMN],
                km_summary[DEFAULT_INTERVAL_END_COLUMN],
            ]).dropna().unique()
            unique_event_times = sorted(unique_times.tolist())

        km_event_table_json = _compute_km_event_table_json(
            km_summary,
            unique_event_times=unique_event_times,
        )

        km_payload = {
            "unique_event_times": unique_event_times,
            "km_event_table": km_event_table_json,
        }

    # 3) Train LR
    init_attrs = {
        "coef_": [[0.0 for _ in predictors]],
        "intercept_": [0.0],
        "classes_": [0, 1],
    }

    lr_result = _logistic_regression_partial(
        km_summary,
        model_attributes=init_attrs,
        predictors=predictors,
        outcome=outcome,
        n_local_iterations=n_local_iterations,
        **model_kwargs,
    )

    # 4) Return combined result
    return {
        **lr_result,
        "km": km_payload,
        "predictors_used": predictors,
    }

# Undecorated version of KM partial for local use and testing; can be called from the decorated version or directly in tests
def _compute_km_event_table_json(km_df: pd.DataFrame, unique_event_times: List[float]) -> str:
    # event table indexed by time grid
    event_table = (
        pd.DataFrame(index=sorted(unique_event_times))
        .rename_axis(DEFAULT_INTERVAL_START_COLUMN)
        .reset_index()
    )

    # Exact events counted at interval_start
    exact_events = km_df[km_df[DEFAULT_EVENT_INDICATOR_COLUMN] == "exact"]
    event_counts = (
        exact_events[DEFAULT_INTERVAL_START_COLUMN]
        .value_counts()
        .reindex(unique_event_times, fill_value=0)
    )

    # Right-censored at interval_start
    censored_events = km_df[km_df[DEFAULT_EVENT_INDICATOR_COLUMN] == "censored"]
    censored_counts = (
        censored_events[DEFAULT_INTERVAL_START_COLUMN]
        .value_counts()
        .reindex(unique_event_times, fill_value=0)
    )

    # Interval events counted at interval_end
    interval_events = km_df[km_df[DEFAULT_EVENT_INDICATOR_COLUMN] == "interval"]
    interval_counts = (
        interval_events[DEFAULT_INTERVAL_END_COLUMN]
        .value_counts()
        .reindex(unique_event_times, fill_value=0)
    )

    total_removed = event_counts + censored_counts + interval_counts

    event_table["removed"] = total_removed.values
    event_table["observed"] = event_counts.values
    event_table["interval"] = interval_counts.values
    event_table["censored"] = censored_counts.values
    event_table["at_risk"] = event_table["removed"].iloc[::-1].cumsum().iloc[::-1]

    return event_table.to_json()

# ---------- Vantage6-exposed partials ----------
@data(1)
def get_unique_event_times_partial(
    df: pd.DataFrame,
    noise_type=None,
    snr: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> List[float]:
    # preprocess to patient-level KM input
    km_df = strata_fit_data_to_km_input(df)

    # unique times from standardized columns
    unique_times = pd.concat([
        km_df[DEFAULT_INTERVAL_START_COLUMN],
        km_df[DEFAULT_INTERVAL_END_COLUMN],
    ]).dropna().unique()

    return sorted(unique_times.tolist())

@data(1)
def validate_partial(
    df: pd.DataFrame,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate local dataframe against configured schema."""
    models = load_data_models_from_settings()
    target = model_name or next(iter(models.keys()))
    model = models[target]
    _, errors = validate_csv(df, model)
    total_rows = len(df.index)
    total_errors = len(errors)
    return {
        "total_rows": total_rows,
        "total_errors": total_errors,
        "error_rate_per_row": (total_errors / total_rows) if total_rows else 0,
        "validation_passed": total_errors == 0,
    }


@data(1)
def imputation_compute_partial(
    df: pd.DataFrame,
    columns: List[str],
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER,
) -> Dict:
    """Compute node-level imputation metrics."""
    imputer_cls = STRATEGY_REGISTRY[imputation_strategy]
    imputer = imputer_cls()
    return imputer.compute(df, columns).to_dict()


@data(1)
def impute_and_train_lr(
    df: pd.DataFrame,
    global_metrics: Dict[str, Any],
    predictors: List[str],
    outcome: str,
    n_local_iterations: int = 50,
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER,
    run_km: bool = True,
    unique_event_times: Optional[List[float]] = None,
    km_noise_type=None,
    km_snr: Optional[float] = None,
    km_random_seed: Optional[int] = None,
    **model_kwargs,
) -> Dict[str, Any]:
    """
    Impute locally using provided global metrics, optionally compute KM on imputed data,
    then fit logistic regression.
    """
    return _impute_and_train_lr_core(
        df,
        global_metrics=global_metrics,
        predictors=predictors,
        outcome=outcome,
        n_local_iterations=n_local_iterations,
        imputation_strategy=imputation_strategy,
        run_km=run_km,
        unique_event_times=unique_event_times,
        km_noise_type=km_noise_type,
        km_snr=km_snr,
        km_random_seed=km_random_seed,
        **model_kwargs,
    )
