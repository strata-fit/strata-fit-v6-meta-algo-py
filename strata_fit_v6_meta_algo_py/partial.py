"""
Local helpers used by the meta‑algorithm.
These are kept decorator‑free so they can be reused inside other V6 entrypoints or tests.
"""
from typing import Any, Dict, List
import pandas as pd

from strata_fit_v6_imputation_py.imputation_strategies.mean import MeanImputer


def impute_locally(df: pd.DataFrame, global_metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply global mean metrics to a local dataframe.
    """
    imputer = MeanImputer()
    return imputer.impute(df, global_metrics)
