"""Preprocessing adapters for STRATA-FIT raw data in the meta algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

import numpy as np
import pandas as pd


class EventType(str, Enum):
    EXACT = "exact"
    CENSORED = "censored"
    INTERVAL = "interval"


DEFAULT_INTERVAL_START_COLUMN = "interval_start"
DEFAULT_INTERVAL_END_COLUMN = "interval_end"
DEFAULT_EVENT_INDICATOR_COLUMN = "event_type"
DEFAULT_EVENT_DEFINITION = "d2t_ra_v2026_selected_v1"


@dataclass(frozen=True)
class D2TDefinitionConfig:
    identifier: str
    label: str
    min_moa: int = 2
    require_activity: bool = True
    use_rolling_activity: bool = True
    include_crp_activity: bool = True
    require_problematic: bool = True
    vas_threshold: float = 50.0
    require_time_since_last_dmard: bool = True
    activity_after_dmard_window: bool = True
    within_years_from_diagnosis: int | None = None


DEFINITION_SENSITIVITY_CONFIGS: tuple[D2TDefinitionConfig, ...] = (
    D2TDefinitionConfig(
        identifier="d2t_candidate_1_moa2",
        label=">=2 b/tsDMARD MOAs",
        require_activity=False,
        include_crp_activity=False,
        require_problematic=False,
    ),
    D2TDefinitionConfig(
        identifier="d2t_candidate_2_moa2_das28",
        label=">=2 MOAs + DAS28 >= 3.2",
        use_rolling_activity=False,
        include_crp_activity=False,
        require_problematic=False,
    ),
    D2TDefinitionConfig(
        identifier="d2t_candidate_3_moa2_das28_vas50",
        label=">=2 MOAs + DAS28 >= 3.2 + VAS >= 50",
        use_rolling_activity=False,
        include_crp_activity=False,
    ),
    D2TDefinitionConfig(
        identifier=DEFAULT_EVENT_DEFINITION,
        label="Selected definition: >=2 MOAs + rolling DAS28/CRP + VAS >= 50",
    ),
    D2TDefinitionConfig(
        identifier="d2t_candidate_5_moa2_rolling_vas75",
        label=">=2 MOAs + rolling DAS28/CRP + VAS >= 75",
        vas_threshold=75.0,
    ),
    D2TDefinitionConfig(
        identifier="d2t_candidate_6_moa3",
        label=">=3 MOAs + rolling DAS28/CRP + VAS >= 75",
        min_moa=3,
        require_time_since_last_dmard=False,
        vas_threshold=75.0,
    ),
    D2TDefinitionConfig(
        identifier="d2t_candidate_7_moa2_within10y",
        label=">=2 MOAs within 10 years + rolling DAS28/CRP + VAS >= 75",
        within_years_from_diagnosis=10,
        vas_threshold=75.0,
    ),
    D2TDefinitionConfig(
        identifier="d2t_candidate_8_moa2_within5y",
        label=">=2 MOAs within 5 years + rolling DAS28/CRP + VAS >= 75",
        within_years_from_diagnosis=5,
        vas_threshold=75.0,
    ),
)

_DEFINITION_CONFIGS = {config.identifier: config for config in DEFINITION_SENSITIVITY_CONFIGS}


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_unique_dmards(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values(["pat_ID", "Visit_months_from_diagnosis"]).copy()
    df["tsDMARD_binary"] = df["tsDMARD"].apply(
        lambda value: np.nan if pd.isna(value) else (1 if float(value) > 0 else 0)
    )
    prev_b = _safe_numeric(df["N_prev_bDMARD"]) if "N_prev_bDMARD" in df.columns else pd.Series(0.0, index=df.index)
    prev_t = _safe_numeric(df["N_prev_tsDMARD"]) if "N_prev_tsDMARD" in df.columns else pd.Series(0.0, index=df.index)
    current_b = _safe_numeric(df["bDMARD"]).fillna(0).gt(0).astype(int)
    current_t = df["tsDMARD_binary"].fillna(0).gt(0).astype(int)
    df["history_moa_floor"] = prev_b.fillna(0) + prev_t.fillna(0) + current_b + current_t

    values: dict[int, int] = {}
    for _, sub_df in df.groupby("pat_ID", sort=False):
        seen: set[tuple[str, int]] = set()
        for idx, b_dmard, ts_dmard, history_floor in zip(
            sub_df.index,
            sub_df["bDMARD"],
            sub_df["tsDMARD_binary"],
            sub_df["history_moa_floor"],
        ):
            if pd.notna(b_dmard) and float(b_dmard) > 0:
                seen.add(("b", int(b_dmard)))
            if pd.notna(ts_dmard) and float(ts_dmard) > 0:
                seen.add(("t", 1))
            values[idx] = max(len(seen), int(history_floor))

    return pd.Series(values).reindex(df.index)


def _has_csdmard_exposure(row: pd.Series) -> bool:
    if "N_prev_csDMARD" in row.index and pd.notna(row["N_prev_csDMARD"]) and float(row["N_prev_csDMARD"]) > 0:
        return True
    for column in ("csDMARD1", "csDMARD2", "csDMARD3"):
        if column in row.index and pd.notna(row[column]) and float(row[column]) > 0:
            return True
    return False


def _first_non_null(series: pd.Series) -> float | int | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return valid.iloc[0]


def get_d2t_definition_config(identifier: str = DEFAULT_EVENT_DEFINITION) -> D2TDefinitionConfig:
    try:
        return _DEFINITION_CONFIGS[identifier]
    except KeyError as exc:
        raise ValueError(f"Unsupported event definition: {identifier}") from exc


def list_supported_d2t_definitions() -> list[dict[str, Any]]:
    return [{"id": config.identifier, "label": config.label} for config in DEFINITION_SENSITIVITY_CONFIGS]


def _normalize_year_floor(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["pat_ID", "Visit_months_from_diagnosis"]).copy()
    shift_mask = ordered["Year_diagnosis"] < 2006
    year_shift = 2006 - ordered.loc[shift_mask, "Year_diagnosis"]
    ordered.loc[shift_mask, "Visit_months_from_diagnosis"] = (
        ordered.loc[shift_mask, "Visit_months_from_diagnosis"] - year_shift * 12
    )
    ordered.loc[shift_mask, "Year_diagnosis"] = 2006
    ordered = ordered[ordered["Visit_months_from_diagnosis"] >= 0].reset_index(drop=True)
    ordered["Year_visit"] = ordered["Year_diagnosis"] + (
        ordered["Visit_months_from_diagnosis"] / 12.0
    ).astype(int)
    return ordered


def _apply_patient_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    patient_summary = (
        df.groupby("pat_ID", as_index=False)
        .agg(
            max_followup=("Visit_months_from_diagnosis", "max"),
            latest_year=("Year_visit", "max"),
        )
    )
    eligible_ids = patient_summary[
        (patient_summary["max_followup"] >= 24.0) & (patient_summary["latest_year"] >= 2010)
    ]["pat_ID"]
    return df[df["pat_ID"].isin(set(eligible_ids.tolist()))].copy()


def derive_d2t_ra_visit_flags(
    df: pd.DataFrame,
    *,
    event_definition: str = DEFAULT_EVENT_DEFINITION,
    apply_patient_eligibility: bool = True,
    exclude_prevalent_at_baseline: bool = False,
) -> pd.DataFrame:
    """Return visit-level D2T RA criteria and event flags for a versioned definition."""
    config = get_d2t_definition_config(event_definition)
    ordered = _normalize_year_floor(df)
    if apply_patient_eligibility:
        ordered = _apply_patient_eligibility(ordered)
    if ordered.empty:
        return ordered

    ordered["cum_unique_btsDMARD"] = compute_unique_dmards(ordered)
    ordered["has_csdmard_exposure"] = ordered.apply(_has_csdmard_exposure, axis=1)
    ordered["cum_ever_csDMARD"] = (
        ordered.groupby("pat_ID")["has_csdmard_exposure"].cummax().astype(bool)
    )
    ordered["last_dmard_change_id"] = (
        ordered.groupby("pat_ID")["cum_unique_btsDMARD"].transform(lambda values: values.ne(values.shift()).cumsum())
    )
    ordered["last_dmard_start_month"] = (
        ordered.groupby(["pat_ID", "last_dmard_change_id"])["Visit_months_from_diagnosis"].transform("min")
    )
    ordered["months_since_last_dmard"] = (
        ordered["Visit_months_from_diagnosis"] - ordered["last_dmard_start_month"]
    )
    ordered["rolling_avg_DAS28"] = (
        ordered.groupby("pat_ID")["DAS28"].rolling(window=2, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    ordered["rolling_avg_CRP"] = (
        ordered.groupby("pat_ID")["CRP"].rolling(window=2, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    criterion1 = ordered["cum_ever_csDMARD"] & (ordered["cum_unique_btsDMARD"] >= config.min_moa)
    if config.require_time_since_last_dmard:
        criterion1 = criterion1 & (ordered["months_since_last_dmard"] >= 6.0)
    if config.within_years_from_diagnosis is not None:
        criterion1 = criterion1 & (
            ordered["Visit_months_from_diagnosis"] <= float(config.within_years_from_diagnosis * 12)
        )

    activity_eligible = (ordered["months_since_last_dmard"] >= 6.0) if config.activity_after_dmard_window else True
    if config.require_activity:
        if config.use_rolling_activity:
            das28_active = ordered["rolling_avg_DAS28"] >= 3.2
            crp_active = ordered["rolling_avg_CRP"] > 10.0 if config.include_crp_activity else False
        else:
            das28_active = ordered["DAS28"] >= 3.2
            crp_active = ordered["CRP"] > 10.0 if config.include_crp_activity else False
        criterion2 = activity_eligible & (das28_active | crp_active)
    else:
        criterion2 = pd.Series(True, index=ordered.index)

    if config.require_problematic:
        criterion3 = (
            (_safe_numeric(ordered["Pat_global"]) >= config.vas_threshold)
            | (_safe_numeric(ordered["Ph_global"]) >= config.vas_threshold)
        )
    else:
        criterion3 = pd.Series(True, index=ordered.index)

    ordered["D2T_crit1"] = criterion1.astype(bool)
    ordered["D2T_crit2"] = criterion2.astype(bool)
    ordered["D2T_crit3"] = criterion3.astype(bool)
    ordered["D2T_RA"] = ordered["D2T_crit1"] & ordered["D2T_crit2"] & ordered["D2T_crit3"]
    ordered["event_definition"] = config.identifier

    if exclude_prevalent_at_baseline:
        baseline_moa = ordered.groupby("pat_ID")["cum_unique_btsDMARD"].transform("first")
        ordered = ordered[baseline_moa < (config.min_moa + 1)].copy()

    return ordered.reset_index(drop=True)


def _build_patient_summary_for_filtering(
    df: pd.DataFrame,
    *,
    event_definition: str = DEFAULT_EVENT_DEFINITION,
) -> pd.DataFrame:
    tagged = derive_d2t_ra_visit_flags(
        df,
        event_definition=event_definition,
        apply_patient_eligibility=False,
        exclude_prevalent_at_baseline=False,
    )
    rows: list[dict[str, Any]] = []
    for patient_id, group in tagged.groupby("pat_ID"):
        rows.append(
            {
                "pat_ID": patient_id,
                "sex": _first_non_null(group["Sex"]),
                "rf_positivity": _first_non_null(group["RF_positivity"]),
                "anti_ccp": _first_non_null(group["anti_CCP"]),
                "ever_csDMARD": bool(group["cum_ever_csDMARD"].fillna(False).any()),
                "ever_bDMARD": bool((_safe_numeric(group["bDMARD"]).fillna(0) > 0).any()),
                "ever_tsDMARD": bool((_safe_numeric(group["tsDMARD"]).fillna(0) > 0).any()),
                "ever_GC": bool((_safe_numeric(group["GC"]).fillna(0) > 0).any()),
                "d2t_like_ever": bool(group["D2T_RA"].fillna(False).any()),
            }
        )
    return pd.DataFrame(rows)


def filter_dataframe_for_cohort(
    df: pd.DataFrame,
    cohort: dict[str, Any] | None,
    *,
    event_definition: str = DEFAULT_EVENT_DEFINITION,
) -> pd.DataFrame:
    if not cohort:
        return df.copy()
    patient_df = _build_patient_summary_for_filtering(df, event_definition=event_definition)
    filtered = patient_df.copy()

    population = cohort.get("population", "all_ra")
    if population == "d2t_like":
        filtered = filtered[filtered["d2t_like_ever"]]
    elif population == "non_d2t_like":
        filtered = filtered[~filtered["d2t_like_ever"]]

    sexes = set(cohort.get("sexes", []) or [])
    if sexes == {"female"}:
        filtered = filtered[filtered["sex"] == 1]
    elif sexes == {"male"}:
        filtered = filtered[filtered["sex"] == 0]

    rf_values = set(cohort.get("rf_positivity_values", []) or [])
    if rf_values == {"positive"}:
        filtered = filtered[filtered["rf_positivity"] == 1]
    elif rf_values == {"negative"}:
        filtered = filtered[filtered["rf_positivity"] == 0]

    anti_ccp_values = set(cohort.get("anti_ccp_values", []) or [])
    if anti_ccp_values == {"positive"}:
        filtered = filtered[filtered["anti_ccp"] == 1]
    elif anti_ccp_values == {"negative"}:
        filtered = filtered[filtered["anti_ccp"] == 0]

    for drug_filter in cohort.get("drug_class_filters", []) or []:
        drug_class = drug_filter.get("drug_class")
        codes = [int(code) for code in drug_filter.get("codes", []) if code is not None]
        if drug_class == "csDMARD":
            filtered = filtered[filtered["ever_csDMARD"]]
        elif drug_class == "bDMARD":
            filtered = filtered[filtered["ever_bDMARD"]]
        elif drug_class == "tsDMARD":
            filtered = filtered[filtered["ever_tsDMARD"]]
        elif drug_class == "GC":
            filtered = filtered[filtered["ever_GC"]]
        if not codes or filtered.empty:
            continue

        keep_ids: list[Any] = []
        for patient_id, group in df.groupby("pat_ID"):
            if patient_id not in set(filtered["pat_ID"].tolist()):
                continue
            if drug_class == "csDMARD":
                matched = any((_safe_numeric(group[column]).isin(codes)).fillna(False).any() for column in ("csDMARD1", "csDMARD2", "csDMARD3"))
            elif drug_class == "bDMARD":
                matched = (_safe_numeric(group["bDMARD"]).isin(codes)).fillna(False).any()
            elif drug_class == "tsDMARD":
                matched = (_safe_numeric(group["tsDMARD"]).isin(codes)).fillna(False).any()
            elif drug_class == "GC" and "GC_type" in group.columns:
                matched = (_safe_numeric(group["GC_type"]).isin(codes)).fillna(False).any()
            else:
                matched = False
            if matched:
                keep_ids.append(patient_id)
        filtered = filtered[filtered["pat_ID"].isin(keep_ids)]

    return df[df["pat_ID"].isin(set(filtered["pat_ID"].tolist()))].copy()


def strata_fit_data_to_km_input(
    df: pd.DataFrame,
    *,
    event_definition: str = DEFAULT_EVENT_DEFINITION,
) -> pd.DataFrame:
    tagged = derive_d2t_ra_visit_flags(
        df,
        event_definition=event_definition,
        apply_patient_eligibility=True,
        exclude_prevalent_at_baseline=True,
    )
    if tagged.empty:
        return pd.DataFrame(
            columns=[
                "pat_ID",
                "Year_diagnosis",
                "D2T_RA_Ever",
                "TTE",
                "maxFU",
                DEFAULT_INTERVAL_START_COLUMN,
                DEFAULT_INTERVAL_END_COLUMN,
                DEFAULT_EVENT_INDICATOR_COLUMN,
                "event_definition",
            ]
        )

    summary = (
        tagged.groupby("pat_ID")
        .agg(
            Year_diagnosis=("Year_diagnosis", "first"),
            D2T_RA_Ever=("D2T_RA", "max"),
            TTE=(
                "Visit_months_from_diagnosis",
                lambda series: (
                    float(series[tagged.loc[series.index, "D2T_RA"]].min())
                    if bool(tagged.loc[series.index, "D2T_RA"].any())
                    else np.nan
                ),
            ),
            maxFU=("Visit_months_from_diagnosis", "max"),
        )
        .reset_index()
    )
    summary["D2T_RA_Ever"] = summary["D2T_RA_Ever"].fillna(False).astype(bool)
    summary["TTE"] = summary["TTE"].fillna(summary["maxFU"]).astype(float)
    summary[DEFAULT_INTERVAL_START_COLUMN] = summary["TTE"]
    summary[DEFAULT_INTERVAL_END_COLUMN] = summary["TTE"]
    summary[DEFAULT_EVENT_INDICATOR_COLUMN] = np.where(
        summary["D2T_RA_Ever"],
        EventType.EXACT.value,
        EventType.CENSORED.value,
    )
    summary["event_definition"] = event_definition
    return summary


def strata_fit_data_to_cox_input(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    outcome_col: str = "event",
    expl_vars: Iterable[str] | None = None,
    event_definition: str = DEFAULT_EVENT_DEFINITION,
) -> pd.DataFrame:
    covariates = list(expl_vars or [])
    km_summary = strata_fit_data_to_km_input(df, event_definition=event_definition)

    required_km_columns = {"pat_ID", "TTE", DEFAULT_EVENT_INDICATOR_COLUMN}
    missing_km_columns = sorted(required_km_columns - set(km_summary.columns))
    if missing_km_columns:
        raise ValueError(f"KM preprocessing output missing columns: {missing_km_columns}")

    # Keep the original diagnosis year for the <2006 reference category.
    visits = df.sort_values(["pat_ID", "Visit_months_from_diagnosis"]).copy()
    visits = visits[visits["pat_ID"].isin(set(km_summary["pat_ID"].tolist()))].copy()
    visits = derive_paper_cox_covariates(visits)
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
        cox_df[DEFAULT_EVENT_INDICATOR_COLUMN] == EventType.EXACT.value
    ).astype(int)
    selected = ["pat_ID", time_col, outcome_col, *covariates]
    return cox_df[selected]


def derive_paper_cox_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Add the pre-specified categorical Cox contrasts as numeric indicators."""
    out = df.copy()
    sex = _safe_numeric(out["Sex"])
    rf = _safe_numeric(out["RF_positivity"])
    anti_ccp = _safe_numeric(out["anti_CCP"])
    year = _safe_numeric(out["Year_diagnosis"])
    out["Sex_Female"] = np.where(sex.isna(), np.nan, (sex == 1).astype(float))
    missing = (
        out["_serology_missing_original"].fillna(False).astype(bool)
        if "_serology_missing_original" in out.columns
        else rf.isna() | anti_ccp.isna()
    )
    out["Serology_Either"] = ((rf == 1) ^ (anti_ccp == 1)).astype(float)
    out["Serology_Both"] = ((rf == 1) & (anti_ccp == 1)).astype(float)
    out["Serology_Missing"] = missing.astype(float)
    out.loc[missing, ["Serology_Either", "Serology_Both"]] = 0.0
    for name, low, high in (
        ("Diagnosis_year_2006_2010", 2006, 2010),
        ("Diagnosis_year_2011_2015", 2011, 2015),
        ("Diagnosis_year_2016_2024", 2016, 2024),
    ):
        out[name] = year.between(low, high, inclusive="both").astype(float)
        out.loc[year.isna(), name] = np.nan
    return out


def compute_d2t_characteristics_components(
    df: pd.DataFrame, *, event_definition: str = DEFAULT_EVENT_DEFINITION
) -> dict[str, float | int]:
    """Return aggregate-only components at each patient's first D2T visit."""
    tagged = derive_d2t_ra_visit_flags(
        df, event_definition=event_definition,
        apply_patient_eligibility=True, exclude_prevalent_at_baseline=False,
    )
    first = (tagged[tagged["D2T_RA"]]
             .sort_values(["pat_ID", "Visit_months_from_diagnosis"])
             .drop_duplicates("pat_ID", keep="first"))

    def components(column: str, prefix: str) -> dict[str, float | int]:
        values = _safe_numeric(first[column])
        return {f"{prefix}_count": int(values.notna().sum()),
                f"{prefix}_sum": float(values.fillna(0).sum()),
                f"{prefix}_sum_sq": float(values.fillna(0).pow(2).sum())}

    result: dict[str, float | int] = {"d2t_patients": int(len(first))}
    for column, prefix in (("Sex", "female"), ("RF_positivity", "rf"),
                           ("anti_CCP", "anti_ccp"), ("Age_diagnosis", "age"),
                           ("DAS28", "das28")):
        result.update(components(column, prefix))
    return result


def compute_d2t_prevalence_by_year(
    df: pd.DataFrame,
    *,
    event_definition: str = DEFAULT_EVENT_DEFINITION,
) -> pd.DataFrame:
    tagged = derive_d2t_ra_visit_flags(
        df,
        event_definition=event_definition,
        apply_patient_eligibility=True,
        exclude_prevalent_at_baseline=False,
    )
    if tagged.empty:
        return pd.DataFrame(
            columns=["Year_visit", "total_patients", "d2t_positive", "partial_year", "max_month_observed"]
        )

    patient_year = (
        tagged.groupby(["Year_visit", "pat_ID"], as_index=False)
        .agg(
            d2t_positive=("D2T_RA", "max"),
            max_month_in_year=("Visit_months_from_diagnosis", lambda values: float((values % 12).max()) if len(values) else np.nan),
        )
    )
    yearly = (
        patient_year.groupby("Year_visit", as_index=False)
        .agg(
            total_patients=("pat_ID", "nunique"),
            d2t_positive=("d2t_positive", "sum"),
            max_month_observed=("max_month_in_year", "max"),
        )
    )
    latest_year = int(yearly["Year_visit"].max())
    yearly["partial_year"] = (
        (yearly["Year_visit"] == latest_year) & (yearly["max_month_observed"] < 11.0)
    )
    return yearly
