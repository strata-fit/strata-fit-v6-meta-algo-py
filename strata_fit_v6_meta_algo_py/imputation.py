from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Type

import numpy as np
import pandas as pd


class ImputationStrategyEnum(str, Enum):
    MEAN_IMPUTER = "mean"
    MEDIAN_IMPUTER = "median"
    CONSTANT_IMPUTER = "constant"
    MICE_IMPUTER = "mice"


class ImputationStrategy(ABC):
    @abstractmethod
    def compute(
        self,
        df: pd.DataFrame,
        columns: List[str],
        global_state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def aggregate(
        self,
        node_metrics: List[Dict[Any, Any]],
        columns: List[str],
        global_means: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def impute(self, df: pd.DataFrame, global_metric: Dict[str, Any]) -> Dict[str, Any]:
        pass


STRATEGY_REGISTRY: Dict[ImputationStrategyEnum, Type[ImputationStrategy]] = {}


def register_imputation_strategy(key: ImputationStrategyEnum):
    def decorator(cls: Type[ImputationStrategy]) -> Type[ImputationStrategy]:
        if not issubclass(cls, ImputationStrategy):
            raise TypeError(
                f"{cls.__name__} must inherit from ImputationStrategy to register under {key}"
            )
        STRATEGY_REGISTRY[key] = cls
        return cls

    return decorator


def _grouped_rows(df: pd.DataFrame, columns: List[str], reducer: str) -> pd.DataFrame:
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.DataFrame(columns=[*columns, "n"])

    if "pat_ID" in df.columns:
        grouped = df.groupby("pat_ID", dropna=False)[available]
        reduced = grouped.mean() if reducer == "mean" else grouped.median()
        counts = df.groupby("pat_ID", dropna=False).size().rename("n")
        result = reduced.join(counts).reset_index()
    else:
        values: Dict[str, Any] = {}
        for column in available:
            series = pd.to_numeric(df[column], errors="coerce")
            values[column] = float(series.mean()) if reducer == "mean" else float(series.median())
        values["n"] = int(len(df.index))
        result = pd.DataFrame([values])

    for column in columns:
        if column not in result.columns:
            result[column] = np.nan
    return result


def _stack_metric_rows(results: List[Dict[Any, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for payload in results:
        if not isinstance(payload, dict):
            continue
        materialized = {
            column: list(inner.values()) if isinstance(inner, dict) else [inner]
            for column, inner in payload.items()
        }
        if materialized:
            frames.append(pd.DataFrame(materialized))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _single_row_metric(values: Dict[str, float]) -> Dict[str, Dict[int, float]]:
    return {column: {0: float(value)} for column, value in values.items()}


def _extract_scalar_metric(global_metric: Dict[str, Any], columns: List[str]) -> Dict[str, float]:
    extracted: Dict[str, float] = {}
    for column in columns:
        value = global_metric.get(column)
        if isinstance(value, dict):
            if value:
                extracted[column] = float(next(iter(value.values())))
        elif value is not None:
            extracted[column] = float(value)
    return extracted


def _build_imputation_model_config(
    strategy: str,
    *,
    columns: List[str],
    state: Dict[str, Any],
    max_iter: int = 5,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "type": "imputation",
        "strategy": strategy,
        "fitted": True,
        "parameters": {
            "columns": list(columns),
            "max_iter": int(max_iter),
        },
        "state": state,
        "metadata": {},
    }


@register_imputation_strategy(ImputationStrategyEnum.MEAN_IMPUTER)
class MeanImputer(ImputationStrategy):
    def compute(
        self,
        df: pd.DataFrame,
        columns: List[str],
        global_state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        del global_state
        return _grouped_rows(df, columns, reducer="mean").to_dict()

    def aggregate(
        self,
        node_metrics: List[Dict[Any, Any]],
        columns: List[str],
        global_means: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        del global_means
        metrics = _stack_metric_rows(node_metrics)
        if metrics.empty:
            return _single_row_metric({column: 0.0 for column in columns})

        weights = pd.to_numeric(metrics.get("n"), errors="coerce").fillna(0.0)
        total_weight = float(weights.sum())
        aggregated: Dict[str, float] = {}
        for column in columns:
            series = pd.to_numeric(metrics.get(column), errors="coerce")
            if series is None or total_weight <= 0.0 or series.dropna().empty:
                aggregated[column] = 0.0
                continue
            aggregated[column] = float((series.fillna(0.0) * weights).sum() / total_weight)
        return _single_row_metric(aggregated)

    def impute(self, df: pd.DataFrame, global_metric: Dict[str, Any]) -> Dict[str, Any]:
        impute_values = _extract_scalar_metric(global_metric, list(df.columns))
        return df.fillna(impute_values).to_dict()


@register_imputation_strategy(ImputationStrategyEnum.MEDIAN_IMPUTER)
class MedianImputer(ImputationStrategy):
    def compute(
        self,
        df: pd.DataFrame,
        columns: List[str],
        global_state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        del global_state
        return _grouped_rows(df, columns, reducer="median").to_dict()

    def aggregate(
        self,
        node_metrics: List[Dict[Any, Any]],
        columns: List[str],
        global_means: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        del global_means
        metrics = _stack_metric_rows(node_metrics)
        aggregated: Dict[str, float] = {}
        for column in columns:
            series = pd.to_numeric(metrics.get(column), errors="coerce")
            aggregated[column] = float(series.median()) if series is not None and not series.dropna().empty else 0.0
        return _single_row_metric(aggregated)

    def impute(self, df: pd.DataFrame, global_metric: Dict[str, Any]) -> Dict[str, Any]:
        impute_values = _extract_scalar_metric(global_metric, list(df.columns))
        return df.fillna(impute_values).to_dict()


@register_imputation_strategy(ImputationStrategyEnum.CONSTANT_IMPUTER)
class ConstantImputer(ImputationStrategy):
    def compute(
        self,
        df: pd.DataFrame,
        columns: List[str],
        global_state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        del df, global_state
        return _single_row_metric({column: 0.0 for column in columns})

    def aggregate(
        self,
        node_metrics: List[Dict[Any, Any]],
        columns: List[str],
        global_means: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        del node_metrics, global_means
        return _single_row_metric({column: 0.0 for column in columns})

    def impute(self, df: pd.DataFrame, global_metric: Dict[str, Any]) -> Dict[str, Any]:
        impute_values = _extract_scalar_metric(global_metric, list(df.columns))
        if not impute_values:
            impute_values = {column: 0.0 for column in df.columns}
        return df.fillna(impute_values).to_dict()


@register_imputation_strategy(ImputationStrategyEnum.MICE_IMPUTER)
class MiceImputer(ImputationStrategy):
    """
    Copied from strata-fit-v6-imputation-py
    (main @ 09de91d, PR #11 "Fix/change imputation loop and column order").
    Only lines marked ``META-ALGO`` differ from the original.

    Federated deterministic MICE (pooled OLS/ridge per feature) that follows
    the algorithm of sklearn's ``IterativeImputer`` (imputation_order="ascending").

    Model state
    -----------
    ``initial_means``      global per-column means (used for the initial fill)
    ``feature_order``      order in which the columns are visited in every
                           iteration (ascending fraction of missing values)
    ``estimate_sequence``  ALL fitted regressions in the order they were fitted:
                           ``max_iter * n_columns`` entries, each with
                           ``iteration``, ``feat_idx``, ``neighbor_indices``,
                           ``coef`` and ``intercept``

    The working matrix of a node is never stored. It is reconstructed by
    *replaying* the estimate sequence on the mean-filled local data
    (``_replay``). This reproduces the persistent state of the sequential
    chained equations: the regression for feature ``t`` in iteration ``k`` is fitted
    on exactly the imputations produced by all earlier regressions, and the final
    imputation replays the complete sequence once.

    Only sufficient statistics (X'X, X'y, n) and regression coefficients leave
    a node/the aggregator - never row-level data.

    Backward compatibility: models created by older versions only contain
    ``global_estimates`` (one final estimate per feature); ``impute`` and
    ``compute`` still accept them (legacy behaviour).
    """

    RIDGE = 1e-6

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def imputation_order(
        sum_results: List[Dict[str, Dict[str, float | int]]],
        columns: List[str],
    ) -> List[int]:
        """
        Order of the features: ascending fraction of missing values, i.e.
        descending number of observed values (all columns share the same number
        of rows). Ties keep the column order (stable), like sklearn.
        """
        counts = np.array(
            [
                sum(int(node.get(column, {}).get("count", 0)) for node in sum_results)
                for column in columns
            ]
        )
        return [int(i) for i in np.argsort(-counts, kind="stable")]

    @staticmethod
    def _sequence(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        sequence = state.get("estimate_sequence")
        if sequence is not None:
            return list(sequence)
        # legacy state: one final estimate per feature, applied in column order
        return sorted(
            state.get("global_estimates", []),
            key=lambda estimate: estimate["feat_idx"],
        )

    @staticmethod
    def _replay(
        df: pd.DataFrame,
        columns: List[str],
        state: Dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Rebuild the current working matrix: mean fill, then apply every stored
        regression in the order it was fitted (only to the originally missing cells).
        Returns (working matrix, boolean mask of originally missing cells).
        """
        data = df[columns].to_numpy(dtype=float, na_value=np.nan)
        missing = np.isnan(data)

        initial_means = state.get("initial_means", {})
        means = np.array([float(initial_means.get(c, 0.0)) for c in columns])
        work = np.where(missing, means, data)

        for estimate in MiceImputer._sequence(state):
            target = estimate["feat_idx"]
            rows = missing[:, target]
            if not rows.any():
                continue
            predictors = estimate["neighbor_indices"]
            coef = np.asarray(estimate["coef"], dtype=float)
            work[rows, target] = (
                float(estimate["intercept"]) + work[rows][:, predictors] @ coef
            )

        return work, missing

    # ============================================================
    # LOCAL PARTIAL COMPUTE
    # ============================================================

    def compute(
        self,
        df: pd.DataFrame,
        columns: List[str],
        global_state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        state = global_state or {}

        target_feat_idx = state.get("target_feat_idx")

        if target_feat_idx is None:
            # META-ALGO: the standalone repo has a separate ``get_local_sums`` partial
            # method for round 0. Here the same payload is returned by
            # ``imputation_compute_partial`` when no target feature is given, so no
            # extra entry point is needed.
            return get_local_sums(df, columns)

        # current working matrix = replay of all estimates fitted so far
        work, missing = self._replay(df, columns, state)

        observed_mask = ~missing[:, target_feat_idx]

        if not observed_mask.any():
            return {"column_stats": []}

        predictor_indices = [
            j for j in range(len(columns)) if j != target_feat_idx
        ]

        X = work[observed_mask][:, predictor_indices]
        y = work[observed_mask, target_feat_idx]

        X = np.hstack([np.ones((X.shape[0], 1)), X])

        xtx = X.T @ X
        xty = X.T @ y

        return {
            "column_stats": [
                {
                    "feat_idx": target_feat_idx,
                    "xtx": xtx.tolist(),
                    "xty": xty.tolist(),
                    "n_obs": int(len(y)),
                }
            ]
        }

    # ============================================================
    # GLOBAL AGGREGATION
    # ============================================================

    def aggregate(
        self,
        node_metrics: List[Dict[str, Any]],
        columns: List[str],
        global_means: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        del global_means

        # only ONE feature processed each call
        feat_idx = None

        global_xtx = None
        global_xty = None

        for node in node_metrics:

            stats = node.get("column_stats", [])

            if not stats:
                continue

            stat = stats[0]

            feat_idx = stat["feat_idx"]

            xtx_local = np.array(stat["xtx"], dtype=float)
            xty_local = np.array(stat["xty"], dtype=float)

            if global_xtx is None:
                global_xtx = xtx_local
                global_xty = xty_local
            else:
                global_xtx += xtx_local
                global_xty += xty_local

        if global_xtx is None:
            return {"global_estimates": []}

        # pooled ridge solve
        global_xtx += (
            np.eye(global_xtx.shape[0]) * self.RIDGE
        )

        beta = np.linalg.solve(global_xtx, global_xty)

        return {
            "global_estimates": [
                {
                    "feat_idx": feat_idx,
                    "neighbor_indices": [
                        i
                        for i in range(len(columns))
                        if i != feat_idx
                    ],
                    "coef": beta[1:].tolist(),
                    "intercept": float(beta[0]),
                }
            ]
        }

    # ============================================================
    # FINAL IMPUTATION
    # ============================================================

    def impute(
        self,
        df: pd.DataFrame,
        global_metric: Dict[str, Any],
    ) -> pd.DataFrame:

        if not global_metric:
            return df.copy()

        model_config = (
            global_metric[0]
            if isinstance(global_metric, list)
            else global_metric
        )

        state = model_config["state"]
        params = model_config["parameters"]

        columns = params["columns"]

        if "estimate_sequence" not in state:
            return self._impute_legacy(df, columns, params, state)

        # replay the complete fitted sequence once
        work, missing = self._replay(df, columns, state)

        result = df.copy()
        for j, column in enumerate(columns):
            if missing[:, j].any():
                result[column] = work[:, j]

        return result

    def _impute_legacy(
        self,
        df: pd.DataFrame,
        columns: List[str],
        params: Dict[str, Any],
        state: Dict[str, Any],
    ) -> pd.DataFrame:
        """Models from older versions: final estimates re-applied max_iter times."""

        max_iter = int(params.get("max_iter", 10))
        initial_means = state["initial_means"]

        data_work = df.copy()

        for col in columns:
            data_work[col] = data_work[col].fillna(float(initial_means[col]))

        estimates = sorted(
            state["global_estimates"],
            key=lambda x: x["feat_idx"],
        )

        for _ in range(max_iter):

            for estimate in estimates:

                target_column = columns[estimate["feat_idx"]]

                missing_mask = df[target_column].isna()

                if not missing_mask.any():
                    continue

                predictor_columns = [
                    columns[i] for i in estimate["neighbor_indices"]
                ]

                X_missing = data_work.loc[
                    missing_mask,
                    predictor_columns,
                ].values

                X_missing = np.hstack(
                    [np.ones((X_missing.shape[0], 1)), X_missing]
                )

                beta = np.array(
                    [estimate["intercept"]] + estimate["coef"],
                    dtype=float,
                )

                data_work.loc[missing_mask, target_column] = X_missing @ beta

        return data_work


# ============================================================
# ROUND 0 + CENTRAL LOOP
# Copied from strata-fit-v6-imputation-py methods.py (get_local_sums_handler,
# _compute_global_means and the MICE branch of central_handler). META-ALGO: the
# vantage6 task creation is replaced by the ``run_partial`` callable so the same
# loop runs with the meta-algo TaskRunner, the in-process client and in tests.
# ============================================================

DEFAULT_MICE_MAX_ITER = 5


def get_local_sums(
    df: pd.DataFrame,
    columns: List[str],
) -> Dict[str, Dict[str, float | int]]:
    results: Dict[str, Dict[str, float | int]] = {}
    for column in columns:
        if column in df.columns:
            series = df[column].dropna()
            results[column] = {
                "sum": float(series.sum()),
                "count": int(series.count()),
            }
        else:
            results[column] = {"sum": 0.0, "count": 0}
    return results


def _compute_global_means(
    sum_results: List[Dict[str, Dict[str, float | int]]],
    columns: List[str],
) -> Dict[str, float]:
    global_means: Dict[str, float] = {}
    for column in columns:
        total_sum = sum(
            float(node.get(column, {}).get("sum", 0.0)) for node in sum_results
        )
        total_count = sum(
            int(node.get(column, {}).get("count", 0)) for node in sum_results
        )
        global_means[column] = (total_sum / total_count) if total_count > 0 else 0.0
    return global_means


def fit_mice_model(
    run_partial: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    columns: List[str],
    max_iter: int | None = None,
) -> Dict[str, Any]:
    """
    ``run_partial(global_state)`` runs ``imputation_compute_partial`` (strategy
    ``mice``) with that ``global_state`` on every node and returns the node payloads.
    Total partial rounds: ``1 + max_iter * len(columns)``.
    """
    # Round 0: global means for the initial fill + visiting order
    sum_results = run_partial({})
    imputer = MiceImputer()
    max_rounds = max_iter or DEFAULT_MICE_MAX_ITER
    # Visit the columns in ascending order of missingness (like sklearn).
    feature_order = imputer.imputation_order(sum_results, columns)
    global_state: Dict[str, Any] = {
        "initial_means": _compute_global_means(sum_results, columns),
        "feature_order": feature_order,
        # every fitted regression, in fitting order. Nodes rebuild their working
        # matrix by replaying this sequence, so each regression is fitted on the
        # imputations of all earlier ones (persistent chained-equations state).
        "estimate_sequence": [],
    }

    for round_num in range(max_rounds):
        for target_feat_idx in feature_order:
            node_metrics = run_partial(
                {**global_state, "target_feat_idx": target_feat_idx}
            )
            aggregated = imputer.aggregate(
                node_metrics=node_metrics,
                columns=columns,
            )
            estimates = aggregated.get("global_estimates", [])
            if not estimates:
                continue
            global_state["estimate_sequence"].append(
                {**estimates[0], "iteration": round_num}
            )

    # Latest estimate per feature, kept for inspection / backward compatibility.
    latest: Dict[int, Dict[str, Any]] = {}
    for estimate in global_state["estimate_sequence"]:
        latest[estimate["feat_idx"]] = estimate
    final_state = {
        **global_state,
        "global_estimates": [latest[idx] for idx in sorted(latest)],
    }

    return _build_imputation_model_config(
        ImputationStrategyEnum.MICE_IMPUTER.value,
        columns=columns,
        state=final_state,
        max_iter=max_rounds,
    )
