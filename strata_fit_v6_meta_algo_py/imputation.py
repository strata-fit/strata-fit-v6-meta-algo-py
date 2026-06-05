from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Type

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


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
    def compute(
        self,
        df: pd.DataFrame,
        columns: List[str],
        global_state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        state = global_state or {}
        data_work = df[columns].copy()

        means = state.get("initial_means", {})
        for column in columns:
            data_work[column] = pd.to_numeric(data_work[column], errors="coerce").fillna(
                float(means.get(column, 0.0))
            )

        for estimate in state.get("global_estimates", []):
            feat_idx = estimate.get("feat_idx")
            if not isinstance(feat_idx, int) or feat_idx < 0 or feat_idx >= len(columns):
                continue

            target_column = columns[feat_idx]
            missing_mask = df[target_column].isna()
            if not missing_mask.any():
                continue

            neighbor_indices = estimate.get("neighbor_indices", [])
            predictor_columns = [
                columns[idx]
                for idx in neighbor_indices
                if isinstance(idx, int) and 0 <= idx < len(columns)
            ]
            if not predictor_columns:
                continue

            x_missing = data_work.loc[missing_mask, predictor_columns].to_numpy(dtype=float)
            x_missing = np.hstack([np.ones((x_missing.shape[0], 1)), x_missing])
            beta = np.array(
                [estimate.get("intercept", 0.0)] + list(estimate.get("coef", [])),
                dtype=float,
            )
            if beta.shape[0] != x_missing.shape[1]:
                continue

            data_work.loc[missing_mask, target_column] = x_missing @ beta

        initial_sums = {
            column: float(pd.to_numeric(df[column], errors="coerce").sum(skipna=True))
            for column in columns
        }
        initial_counts = {
            column: int(pd.to_numeric(df[column], errors="coerce").count())
            for column in columns
        }

        stats_per_column: List[Dict[str, Any]] = []
        for feat_idx, target_column in enumerate(columns):
            observed_mask = df[target_column].notna()
            if not observed_mask.any():
                continue

            y = pd.to_numeric(df.loc[observed_mask, target_column], errors="coerce").to_numpy(dtype=float)
            predictor_columns = [column for column in columns if column != target_column]
            x = data_work.loc[observed_mask, predictor_columns].to_numpy(dtype=float)
            x = np.hstack([np.ones((x.shape[0], 1)), x])

            stats_per_column.append(
                {
                    "feat_idx": feat_idx,
                    "xtx": (x.T @ x).tolist(),
                    "xty": (x.T @ y).tolist(),
                    "n_obs": int(len(y)),
                }
            )

        return {
            "initial_sums": initial_sums,
            "initial_counts": initial_counts,
            "column_stats": stats_per_column,
        }

    def aggregate(
        self,
        node_metrics: List[Dict[str, Any]],
        columns: List[str],
        global_means: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        del global_means

        initial_sums = {column: 0.0 for column in columns}
        initial_counts = {column: 0 for column in columns}
        for node in node_metrics:
            for column in columns:
                initial_sums[column] += float(node.get("initial_sums", {}).get(column, 0.0))
                initial_counts[column] += int(node.get("initial_counts", {}).get(column, 0))

        initial_means = {
            column: (initial_sums[column] / initial_counts[column]) if initial_counts[column] else 0.0
            for column in columns
        }

        global_estimates: List[Dict[str, Any]] = []
        processed_indices = sorted(
            {
                stat.get("feat_idx")
                for node in node_metrics
                for stat in node.get("column_stats", [])
                if isinstance(stat.get("feat_idx"), int)
            }
        )

        for feat_idx in processed_indices:
            global_xtx = None
            global_xty = None

            for node in node_metrics:
                stat = next(
                    (
                        item
                        for item in node.get("column_stats", [])
                        if item.get("feat_idx") == feat_idx
                    ),
                    None,
                )
                if stat is None:
                    continue

                xtx_local = np.array(stat["xtx"], dtype=float)
                xty_local = np.array(stat["xty"], dtype=float)
                global_xtx = xtx_local if global_xtx is None else global_xtx + xtx_local
                global_xty = xty_local if global_xty is None else global_xty + xty_local

            if global_xtx is None or global_xty is None:
                continue

            global_xtx = global_xtx + np.eye(global_xtx.shape[0]) * 1e-6
            beta = np.linalg.solve(global_xtx, global_xty)
            global_estimates.append(
                {
                    "feat_idx": feat_idx,
                    "neighbor_indices": [idx for idx in range(len(columns)) if idx != feat_idx],
                    "coef": beta[1:].tolist(),
                    "intercept": float(beta[0]),
                }
            )

        return _build_imputation_model_config(
            ImputationStrategyEnum.MICE_IMPUTER.value,
            columns=columns,
            state={
                "initial_means": initial_means,
                "global_estimates": global_estimates,
            },
        )

    def impute(self, df: pd.DataFrame, global_metric: Dict[str, Any]) -> Dict[str, Any]:
        if not global_metric:
            return df.to_dict()

        model_config = global_metric[0] if isinstance(global_metric, list) else global_metric
        if not isinstance(model_config, dict):
            return df.to_dict()

        state = model_config.get("state", model_config)
        params = model_config.get("parameters", {})
        columns = params.get("columns")
        if not isinstance(columns, list) or not columns:
            columns = [column for column in df.columns if column in df.columns]
        if not columns:
            return df.to_dict()

        data_work = df.copy()
        initial_means = state.get("initial_means", {})
        for column in columns:
            data_work[column] = pd.to_numeric(data_work[column], errors="coerce").fillna(
                float(initial_means.get(column, 0.0))
            )

        global_estimates = state.get("global_estimates", [])
        for estimate in global_estimates:
            feat_idx = estimate.get("feat_idx")
            if not isinstance(feat_idx, int) or feat_idx < 0 or feat_idx >= len(columns):
                continue

            target_column = columns[feat_idx]
            missing_mask = df[target_column].isna()
            if not missing_mask.any():
                continue

            neighbor_indices = estimate.get("neighbor_indices", [])
            predictor_columns = [
                columns[idx]
                for idx in neighbor_indices
                if isinstance(idx, int) and 0 <= idx < len(columns)
            ]
            if not predictor_columns:
                continue

            estimator = BayesianRidge()
            imputer = IterativeImputer(
                estimator=estimator,
                max_iter=int(params.get("max_iter", 5)),
                random_state=42,
            )
            imputer.fit(data_work[columns].to_numpy(dtype=float))

            x_missing = data_work.loc[missing_mask, predictor_columns].to_numpy(dtype=float)
            beta = np.array(list(estimate.get("coef", [])), dtype=float)
            if beta.shape[0] != x_missing.shape[1]:
                continue

            predictions = x_missing @ beta + float(estimate.get("intercept", 0.0))
            data_work.loc[missing_mask, target_column] = predictions

        data_work[columns] = data_work[columns].fillna(
            {column: float(initial_means.get(column, 0.0)) for column in columns}
        )
        return data_work.to_dict()
