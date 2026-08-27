from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pydantic import BaseModel, Field
from v6_federated_core import MethodContext

from .log import info, warn


class GetUniqueEventTimesInput(BaseModel):
    time_col: str
    outcome_col: str
    minimum_events: int = 10


class GetUniqueEventTimesOutput(BaseModel):
    times: Optional[Dict[str, Dict[Any, Any]]] = None
    n_threshold_not_met: Optional[int] = None


class ComputeSummedZInput(BaseModel):
    outcome_col: str
    expl_vars: List[str] = Field(min_length=1)


class ComputeSummedZOutput(BaseModel):
    sum: Dict[str, float]


class PerformIterationInput(BaseModel):
    time_col: str
    expl_vars: List[str] = Field(min_length=1)
    beta: List[float]
    unique_time_events: List[float]


class PerformIterationOutput(BaseModel):
    agg1: List[float]
    agg2: Dict[str, Dict[Any, float]]
    agg3: List[List[List[float]]]


def _get_dataframe(context: MethodContext) -> pd.DataFrame:
    df = context.meta.get("df")
    if df is None:
        raise RuntimeError("Method context is missing the dataframe")
    return df


def compute_derivatives(
    summed_agg1: np.ndarray,
    summed_agg2: np.ndarray,
    summed_agg3: np.ndarray,
    aggregated_time_events: pd.DataFrame,
    z_sum: pd.Series | np.ndarray | List[float],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(z_sum, pd.Series):
        z_sum_vec = z_sum.to_numpy(dtype=float)
    else:
        z_sum_vec = np.asarray(z_sum, dtype=float)

    n_covariates = z_sum_vec.shape[0]
    tot_p1 = np.zeros(n_covariates, dtype=float)
    tot_p2 = np.zeros((n_covariates, n_covariates), dtype=float)

    for index, row in aggregated_time_events.iterrows():
        denom = float(summed_agg1[index])
        if denom <= 0.0:
            continue
        freq = float(row["freq"])
        s1 = freq * (summed_agg2[index] / denom)
        first_part = summed_agg3[index] / denom
        numerator = np.outer(summed_agg2[index], summed_agg2[index])
        second_part = numerator / (denom * denom)
        s2 = freq * (first_part - second_part)
        tot_p1 += s1
        tot_p2 += s2

    primary_derivative = z_sum_vec - tot_p1
    secondary_derivative = -tot_p2
    return primary_derivative, secondary_derivative


def get_unique_event_times_handler(
    data: GetUniqueEventTimesInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for get_unique_event_times")

    df = _get_dataframe(context)
    client = context.meta.get("client")

    info("Computing unique event times")
    if int(df[data.outcome_col].notnull().sum()) <= int(data.minimum_events):
        org_id = getattr(client, "organization_id", -1)
        warn("Sub-task skipped because the number of samples is too small")
        return {"n_threshold_not_met": int(org_id)}

    times = df[df[data.outcome_col] == 1].groupby(data.time_col, as_index=False).count()
    times = times.sort_values(by=data.time_col)[[data.time_col, data.outcome_col]]
    times["freq"] = times[data.outcome_col]
    times = times.drop(columns=data.outcome_col)
    return {"times": times.to_dict()}


def compute_summed_z_handler(
    data: ComputeSummedZInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for compute_summed_z")

    df = _get_dataframe(context)
    info("Computing summed z statistics")
    z_sum = df[df[data.outcome_col] == 1][data.expl_vars].sum().astype(float).to_dict()
    return {"sum": z_sum}


def perform_iteration_handler(
    data: PerformIterationInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for perform_iteration")

    df = _get_dataframe(context)
    info("Computing aggregates for the derivation of the partial likelihood")

    beta = np.asarray(data.beta, dtype=float)
    working = df[[data.time_col, *data.expl_vars]].dropna(how="any")
    X = working[data.expl_vars].to_numpy(dtype=float)
    times = working[data.time_col].to_numpy(dtype=float)

    if X.shape[0] == 0:
        zeros = np.zeros((len(data.unique_time_events), len(data.expl_vars)))
        return {
            "agg1": [0.0] * len(data.unique_time_events),
            "agg2": pd.DataFrame(zeros, columns=data.expl_vars).to_dict(),
            "agg3": [
                np.zeros((len(data.expl_vars), len(data.expl_vars))).tolist()
                for _ in data.unique_time_events
            ],
        }

    exp_xb = np.exp(X @ beta)
    agg1: List[float] = []
    agg2_rows: List[np.ndarray] = []
    agg3: List[np.ndarray] = []
    n_covariates = len(data.expl_vars)

    for unique_time in data.unique_time_events:
        mask = times >= float(unique_time)
        if not np.any(mask):
            agg1.append(0.0)
            agg2_rows.append(np.zeros(n_covariates, dtype=float))
            agg3.append(np.zeros((n_covariates, n_covariates), dtype=float))
            continue

        Xi = X[mask]
        exp_i = exp_xb[mask]
        weighted = Xi * exp_i[:, None]

        agg1.append(float(exp_i.sum()))
        agg2_rows.append(weighted.sum(axis=0))
        agg3.append(Xi.T @ weighted)

    agg2_df = pd.DataFrame(agg2_rows, columns=data.expl_vars)
    return {
        "agg1": agg1,
        "agg2": agg2_df.to_dict(),
        "agg3": [matrix.tolist() for matrix in agg3],
    }
