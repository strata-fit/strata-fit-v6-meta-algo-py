from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "main",
    "run_local_meta_algorithm",
    "validate_partial",
    "imputation_compute_partial",
    "impute_and_train_sklearn_linear",
    "cox_get_unique_event_times_imputed",
    "cox_compute_summed_z_imputed",
    "cox_perform_iteration_imputed",
    "km_get_unique_event_times_imputed",
    "km_get_event_table_imputed",
    "prevalence_by_year_imputed",
    "cox_risk_score_range_imputed",
    "cox_risk_score_histogram_imputed",
    "cox_risk_group_summary_imputed",
]


if TYPE_CHECKING:
    from .central import main, run_local_meta_algorithm
    from .partial import (
        cox_compute_summed_z_imputed,
        cox_get_unique_event_times_imputed,
        cox_risk_group_summary_imputed,
        cox_risk_score_histogram_imputed,
        cox_risk_score_range_imputed,
        cox_perform_iteration_imputed,
        imputation_compute_partial,
        impute_and_train_sklearn_linear,
        km_get_event_table_imputed,
        km_get_unique_event_times_imputed,
        prevalence_by_year_imputed,
        validate_partial,
    )


def __getattr__(name: str) -> Any:
    if name in {"main", "run_local_meta_algorithm"}:
        from . import central

        return getattr(central, name)

    if name in {
        "validate_partial",
        "imputation_compute_partial",
        "impute_and_train_sklearn_linear",
        "cox_get_unique_event_times_imputed",
        "cox_compute_summed_z_imputed",
        "cox_perform_iteration_imputed",
        "km_get_unique_event_times_imputed",
        "km_get_event_table_imputed",
        "prevalence_by_year_imputed",
        "cox_risk_score_range_imputed",
        "cox_risk_score_histogram_imputed",
        "cox_risk_group_summary_imputed",
    }:
        from . import partial

        return getattr(partial, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
