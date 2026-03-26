"""Preprocessing adapters for STRATA-FIT raw data in the meta algorithm."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from strata_fit_v6_km_py.preprocessing import strata_fit_data_to_km_input as km_preprocess
from strata_fit_v6_km_py.types import (
    DEFAULT_EVENT_INDICATOR_COLUMN,
    DEFAULT_INTERVAL_END_COLUMN,
    DEFAULT_INTERVAL_START_COLUMN,
)


def _first_non_null(series: pd.Series) -> float | int | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return valid.iloc[0]


def strata_fit_data_to_km_input(df: pd.DataFrame) -> pd.DataFrame:
    """Convert STRATA-FIT raw longitudinal records to KM-ready summary rows."""
    return km_preprocess(df.copy())


def strata_fit_data_to_cox_input(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    outcome_col: str = "event",
    expl_vars: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Convert STRATA-FIT raw records to one-row-per-patient Cox input."""
    covariates = list(expl_vars or [])
    km_summary = strata_fit_data_to_km_input(df)

    required_km_columns = {"pat_ID", "TTE", DEFAULT_EVENT_INDICATOR_COLUMN}
    missing_km_columns = sorted(required_km_columns - set(km_summary.columns))
    if missing_km_columns:
        raise ValueError(f"KM preprocessing output missing columns: {missing_km_columns}")

    visits = df.sort_values(["pat_ID", "Visit_months_from_diagnosis"]).copy()
    for covariate in covariates:
        if covariate not in visits.columns:
            raise ValueError(f"Covariate '{covariate}' not found in STRATA-FIT dataframe")

    if covariates:
        baseline = visits.groupby("pat_ID", as_index=False).agg(
            **{
                covariate: pd.NamedAgg(column=covariate, aggfunc=_first_non_null)
                for covariate in covariates
            }
        )
    else:
        baseline = visits[["pat_ID"]].drop_duplicates(ignore_index=True)

    cox_df = km_summary[["pat_ID", "TTE", DEFAULT_EVENT_INDICATOR_COLUMN]].merge(
        baseline,
        on="pat_ID",
        how="left",
    )
    cox_df[time_col] = cox_df["TTE"].astype(float)
    cox_df[outcome_col] = (cox_df[DEFAULT_EVENT_INDICATOR_COLUMN] != "censored").astype(int)

    selected = ["pat_ID", time_col, outcome_col, *covariates]
    return cox_df[selected]
