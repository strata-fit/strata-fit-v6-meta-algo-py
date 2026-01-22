"""
Lightweight meta‑orchestration used for local/mock runs.
This version avoids Vantage6 decorators so it can be called directly in tests.
"""
from typing import Any, Dict, List
import pandas as pd

from strata_fit_v6_imputation_py.imputation_strategies.mean import MeanImputer
from v6_logistic_regression_py.partials import _logistic_regression_partial


def run_pipeline(
    df: pd.DataFrame,
    *,
    imputation_columns: List[str],
    predictors: List[str],
    outcome: str,
    n_local_iterations: int = 50,
) -> Dict[str, Any]:
    """
    1) Compute global mean imputation metrics from the provided dataframe
    2) Impute the dataframe locally
    3) Run a single federated-style LR local fit using the Flower-aware helper

    Returns a dict with imputation metrics and trained LR parameters.
    """
    imputer = MeanImputer()

    node_metric = imputer.compute(df, imputation_columns).to_dict()
    global_metrics = imputer.aggregate([node_metric], imputation_columns)

    imputed_df = imputer.impute(df, global_metrics)

    init_attrs = {
        "coef_": [[0.0 for _ in predictors]],
        "intercept_": [0.0],
        # binary classes for the mock; adjust if more classes are present
        "classes_": [0, 1],
    }

    lr_result = _logistic_regression_partial(
        imputed_df,
        model_attributes=init_attrs,
        predictors=predictors,
        outcome=outcome,
        n_local_iterations=n_local_iterations,
    )

    return {
        "imputation_metrics": global_metrics,
        "lr_model_attributes": lr_result["model_attributes"],
        "training_size": lr_result["size"],
    }


