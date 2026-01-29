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
    **model_kwargs,
) -> Dict[str, Any]:
    imputed = impute_locally(
        df,
        global_metrics,
        imputation_strategy=imputation_strategy,
    )
    init_attrs = {
        "coef_": [[0.0 for _ in predictors]],
        "intercept_": [0.0],
        "classes_": [0, 1],
    }
    return _logistic_regression_partial(
        imputed,
        model_attributes=init_attrs,
        predictors=predictors,
        outcome=outcome,
        n_local_iterations=n_local_iterations,
        **model_kwargs,
    )


# ---------- Vantage6-exposed partials ----------
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
    **model_kwargs,
) -> Dict[str, Any]:
    """
    Impute locally using provided global metrics, then fit logistic regression.
    """
    return _impute_and_train_lr_core(
        df,
        global_metrics=global_metrics,
        predictors=predictors,
        outcome=outcome,
        n_local_iterations=n_local_iterations,
        imputation_strategy=imputation_strategy,
        **model_kwargs,
    )
