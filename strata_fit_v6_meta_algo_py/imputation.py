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


DEFAULT_MICE_MAX_ITER = 5


@register_imputation_strategy(ImputationStrategyEnum.MICE_IMPUTER)
class MiceImputer(ImputationStrategy):
    """
    Federated deterministic MICE (pooled OLS/ridge per feature) that follows the
    algorithm of sklearn's ``IterativeImputer(imputation_order="ascending")``.

    Ported from strata-fit-v6-imputation-py (fix/change-imputation-loop-and-column-order).

    Fitting protocol (orchestrated centrally by ``fit_mice_model``):
      round 0   ``compute`` without ``target_feat_idx`` -> per-column sums/counts;
                ``aggregate`` -> ``initial_means`` + ``feature_order``
      rounds    for every iteration, for every feature in ``feature_order``:
                ``compute`` with ``target_feat_idx`` -> X'X, X'y for that feature
                (on the working matrix rebuilt by replaying ``estimate_sequence``);
                ``aggregate`` -> one pooled regression, appended to the sequence.

    Model state
    -----------
    ``initial_means``      global per-column means (initial fill)
    ``feature_order``      column visiting order (ascending fraction missing)
    ``estimate_sequence``  ALL fitted regressions in fitting order
                           (``max_iter * n_columns`` entries)
    ``global_estimates``   latest estimate per feature (inspection / legacy)

    Only sufficient statistics (X'X, X'y, n, sums, counts) and coefficients leave a
    node - never row-level data.
    """

    RIDGE = 1e-6

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _numeric_matrix(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
        frame = df.reindex(columns=columns)
        return np.column_stack(
            [pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float) for c in columns]
        ) if columns else np.empty((len(df.index), 0))

    @staticmethod
    def imputation_order(counts: Dict[str, int], columns: List[str]) -> List[int]:
        """Descending number of observed values; ties keep column order (stable), like sklearn."""
        observed = np.array([int(counts.get(c, 0)) for c in columns])
        return [int(i) for i in np.argsort(-observed, kind="stable")]

    @staticmethod
    def _sequence(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        sequence = state.get("estimate_sequence")
        if sequence is not None:
            return list(sequence)
        # legacy model: one final estimate per feature, applied once in column order
        return sorted(state.get("global_estimates", []), key=lambda e: e["feat_idx"])

    @classmethod
    def _replay(
        cls,
        df: pd.DataFrame,
        columns: List[str],
        state: Dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean fill, then apply every stored regression in fitting order to the
        originally missing cells. Returns (working matrix, missing mask)."""
        data = cls._numeric_matrix(df, columns)
        missing = np.isnan(data)
        initial_means = state.get("initial_means", {})
        means = np.array([float(initial_means.get(c, 0.0)) for c in columns])
        work = np.where(missing, means, data)

        n_cols = len(columns)
        for estimate in cls._sequence(state):
            target = estimate.get("feat_idx")
            if not isinstance(target, int) or not 0 <= target < n_cols:
                continue
            rows = missing[:, target]
            if not rows.any():
                continue
            predictors = [i for i in estimate.get("neighbor_indices", []) if 0 <= i < n_cols]
            coef = np.asarray(estimate.get("coef", []), dtype=float)
            if not predictors or coef.shape[0] != len(predictors):
                continue
            work[rows, target] = float(estimate.get("intercept", 0.0)) + work[rows][:, predictors] @ coef
        return work, missing

    @staticmethod
    def build_model(columns: List[str], state: Dict[str, Any], max_iter: int) -> Dict[str, Any]:
        latest: Dict[int, Dict[str, Any]] = {}
        for estimate in state.get("estimate_sequence", []):
            latest[estimate["feat_idx"]] = estimate
        final_state = {**state, "global_estimates": [latest[i] for i in sorted(latest)]}
        return _build_imputation_model_config(
            ImputationStrategyEnum.MICE_IMPUTER.value,
            columns=columns,
            state=final_state,
            max_iter=max_iter,
        )

    # ------------------------------------------------------------------ node side

    def compute(
        self,
        df: pd.DataFrame,
        columns: List[str],
        global_state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        state = global_state or {}
        target = state.get("target_feat_idx")

        if target is None:
            # round 0: sufficient statistics for the global means / visiting order
            data = self._numeric_matrix(df, columns)
            observed = ~np.isnan(data)
            return {
                "initial_sums": {c: float(np.nansum(data[:, j])) for j, c in enumerate(columns)},
                "initial_counts": {c: int(observed[:, j].sum()) for j, c in enumerate(columns)},
            }

        target = int(target)
        work, missing = self._replay(df, columns, state)
        observed_mask = ~missing[:, target]
        if not observed_mask.any():
            return {"column_stats": []}

        predictors = [j for j in range(len(columns)) if j != target]
        x = work[observed_mask][:, predictors]
        y = work[observed_mask, target]
        x = np.hstack([np.ones((x.shape[0], 1)), x])
        return {
            "column_stats": [
                {
                    "feat_idx": target,
                    "xtx": (x.T @ x).tolist(),
                    "xty": (x.T @ y).tolist(),
                    "n_obs": int(len(y)),
                }
            ]
        }

    # ------------------------------------------------------------------ central side

    def aggregate(
        self,
        node_metrics: List[Dict[str, Any]],
        columns: List[str],
        global_means: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        del global_means

        if not any("column_stats" in node for node in node_metrics):
            # round 0
            sums = {c: 0.0 for c in columns}
            counts = {c: 0 for c in columns}
            for node in node_metrics:
                for c in columns:
                    sums[c] += float(node.get("initial_sums", {}).get(c, 0.0))
                    counts[c] += int(node.get("initial_counts", {}).get(c, 0))
            return {
                "initial_means": {c: (sums[c] / counts[c]) if counts[c] else 0.0 for c in columns},
                "feature_order": self.imputation_order(counts, columns),
            }

        feat_idx = None
        xtx = None
        xty = None
        for node in node_metrics:
            stats = node.get("column_stats", [])
            if not stats:
                continue
            stat = stats[0]
            feat_idx = stat["feat_idx"]
            xtx_local = np.array(stat["xtx"], dtype=float)
            xty_local = np.array(stat["xty"], dtype=float)
            xtx = xtx_local if xtx is None else xtx + xtx_local
            xty = xty_local if xty is None else xty + xty_local

        if xtx is None:
            return {"global_estimates": []}

        beta = np.linalg.solve(xtx + np.eye(xtx.shape[0]) * self.RIDGE, xty)
        return {
            "global_estimates": [
                {
                    "feat_idx": feat_idx,
                    "neighbor_indices": [i for i in range(len(columns)) if i != feat_idx],
                    "coef": beta[1:].tolist(),
                    "intercept": float(beta[0]),
                }
            ]
        }

    # ------------------------------------------------------------------ apply

    def impute(self, df: pd.DataFrame, global_metric: Dict[str, Any]) -> Dict[str, Any]:
        if not global_metric:
            return df.to_dict()
        model_config = global_metric[0] if isinstance(global_metric, list) else global_metric
        if not isinstance(model_config, dict):
            return df.to_dict()

        state = model_config.get("state", model_config)
        columns = model_config.get("parameters", {}).get("columns")
        if not isinstance(columns, list) or not columns:
            return df.to_dict()

        work, _ = self._replay(df, columns, state)
        result = df.copy()
        for j, column in enumerate(columns):
            result[column] = work[:, j]
        return result.to_dict()


def fit_mice_model(
    run_partial: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    columns: List[str],
    max_iter: int = DEFAULT_MICE_MAX_ITER,
) -> Dict[str, Any]:
    """
    Central MICE fitting loop, independent of the transport.

    ``run_partial(global_state)`` must run ``imputation_compute_partial`` with
    ``strategy="mice"`` and the given ``global_state`` on every node and return the
    list of node payloads. Total partial rounds: ``1 + max_iter * len(columns)``.
    """
    imputer = MiceImputer()
    initial = imputer.aggregate(run_partial({}), columns)
    state: Dict[str, Any] = {**initial, "estimate_sequence": []}

    for iteration in range(int(max_iter)):
        for target in state["feature_order"]:
            node_metrics = run_partial({**state, "target_feat_idx": target})
            estimates = imputer.aggregate(node_metrics, columns).get("global_estimates", [])
            if estimates:
                state["estimate_sequence"].append({**estimates[0], "iteration": iteration})

    return MiceImputer.build_model(columns, state, max_iter=int(max_iter))
