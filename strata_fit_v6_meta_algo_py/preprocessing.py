"""Preprocessing adapters for STRATA-FIT raw data in the meta algorithm."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd


class EventType(str, Enum):
    EXACT = "exact"
    CENSORED = "censored"
    INTERVAL = "interval"


DEFAULT_INTERVAL_START_COLUMN = "interval_start"
DEFAULT_INTERVAL_END_COLUMN = "interval_end"
DEFAULT_EVENT_INDICATOR_COLUMN = "event_type"


def compute_unique_dmards(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values(["pat_ID", "Visit_months_from_diagnosis"]).copy()
    df["tsDMARD_binary"] = df["tsDMARD"].apply(
        lambda value: np.nan if pd.isna(value) else (1 if value != 0 else 0)
    )

    values: dict[int, int] = {}
    for _, sub_df in df.groupby("pat_ID", sort=False):
        seen = set()
        for idx, b_dmard, ts_dmard in zip(sub_df.index, sub_df["bDMARD"], sub_df["tsDMARD_binary"]):
            if not pd.isna(b_dmard):
                seen.add(("b", b_dmard))
            if not pd.isna(ts_dmard):
                seen.add(("t", 1))
            values[idx] = len(seen)

    return pd.Series(values).reindex(df.index)


def _first_non_null(series: pd.Series) -> float | int | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return valid.iloc[0]


def derive_d2t_ra_visit_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return visit-level D2T RA criteria and event flags.

    A visit is a D2T RA event only when all three operational criteria are
    satisfied together. Keeping this in one helper lets tests and validation
    artifacts assert the same logic used by KM/Cox preprocessing.
    """
    df = df.copy()
    df.sort_values(["pat_ID", "Visit_months_from_diagnosis"], inplace=True)

    shift_mask = df["Year_diagnosis"] < 2006
    year_shift = 2006 - df.loc[shift_mask, "Year_diagnosis"]
    df.loc[shift_mask, "Visit_months_from_diagnosis"] = (
        df.loc[shift_mask, "Visit_months_from_diagnosis"] - year_shift * 12
    )
    df.loc[shift_mask, "Year_diagnosis"] = 2006
    df = df[df["Visit_months_from_diagnosis"] >= 0].reset_index(drop=True)

    df["cum_unique_btsDMARD"] = compute_unique_dmards(df)
    df["cum_btsDMARDmin"] = df.groupby("pat_ID")["cum_unique_btsDMARD"].cummin()
    df["last_dmard_change_id"] = (
        df.groupby("pat_ID")["cum_unique_btsDMARD"]
        .transform(lambda series: series.ne(series.shift()).cumsum())
    )
    df["last_dmard_start_month"] = (
        df.groupby(["pat_ID", "last_dmard_change_id"])["Visit_months_from_diagnosis"]
        .transform("min")
    )
    df["months_since_last_dmard"] = (
        df["Visit_months_from_diagnosis"] - df["last_dmard_start_month"]
    )
    df["rolling_avg_DAS28"] = (
        df.groupby("pat_ID")["DAS28"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["rolling_avg_CRP"] = (
        df.groupby("pat_ID")["CRP"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["D2T_crit1"] = (
        (df["cum_unique_btsDMARD"] >= 2) & (df["months_since_last_dmard"] >= 6)
    )
    df["D2T_crit2"] = (
        (df["rolling_avg_DAS28"] > 3.2) | (df["rolling_avg_CRP"] > 1.0)
    )
    df["D2T_crit3"] = (df["Pat_global"] > 50) | (df["Ph_global"] > 50)
    df["D2T_RA"] = df["D2T_crit1"] & df["D2T_crit2"] & df["D2T_crit3"]
    return df


def strata_fit_data_to_km_input(df: pd.DataFrame) -> pd.DataFrame:
    df = derive_d2t_ra_visit_flags(df)

    summary = (
        df.groupby("pat_ID")
        .agg(
            Year_diagnosis=("Year_diagnosis", "first"),
            D2T_RA_Ever=("D2T_RA", "max"),
            cum_btsDMARDmin=("cum_btsDMARDmin", "max"),
            minFU=("Visit_months_from_diagnosis", "min"),
            TTE=(
                "Visit_months_from_diagnosis",
                lambda series: (
                    series[df.loc[series.index, "D2T_RA"]].min()
                    if any(df.loc[series.index, "D2T_RA"])
                    else np.nan
                ),
            ),
            maxFU=("Visit_months_from_diagnosis", "max"),
        )
        .reset_index()
    )

    summary["D2T_RA_Ever"] = summary["D2T_RA_Ever"].fillna(0)
    summary["cum_btsDMARDmin"] = summary["cum_btsDMARDmin"].fillna(0)
    summary["TTE"] = summary["TTE"].fillna(summary["maxFU"])

    summary["cens"] = np.select(
        condlist=[
            (summary["D2T_RA_Ever"] == 1) & (summary["cum_btsDMARDmin"] > 2),
            (summary["D2T_RA_Ever"] == 0),
        ],
        choicelist=["interval", "right"],
        default="no",
    )

    summary[DEFAULT_INTERVAL_START_COLUMN] = np.where(
        summary["cens"] == "interval",
        0,
        summary["TTE"],
    )
    summary[DEFAULT_INTERVAL_END_COLUMN] = np.where(
        summary["cens"] == "interval",
        summary["minFU"],
        summary["TTE"],
    )
    summary[DEFAULT_EVENT_INDICATOR_COLUMN] = np.select(
        condlist=[
            summary["cens"] == "interval",
            summary["cens"] == "no",
        ],
        choicelist=[EventType.INTERVAL.value, EventType.EXACT.value],
        default=EventType.CENSORED.value,
    )
    return summary


def strata_fit_data_to_cox_input(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    outcome_col: str = "event",
    expl_vars: Iterable[str] | None = None,
) -> pd.DataFrame:
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
    cox_df[outcome_col] = (
        cox_df[DEFAULT_EVENT_INDICATOR_COLUMN] != EventType.CENSORED.value
    ).astype(int)

    selected = ["pat_ID", time_col, outcome_col, *covariates]
    return cox_df[selected]
