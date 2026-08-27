from __future__ import annotations

import pandas as pd
import pytest

from strata_fit_v6_meta_algo_py.preprocessing import (
    compute_d2t_prevalence_by_year,
    derive_d2t_ra_visit_flags,
    strata_fit_data_to_cox_input,
    strata_fit_data_to_km_input,
)


def _visit(
    patient_id: str,
    month: float,
    *,
    year_diagnosis: int = 2012,
    bdmard: int | None = 1,
    tsdmard: int | None = None,
    n_prev_bdmard: int = 0,
    n_prev_tsdmard: int = 0,
    das28: float | None = 2.4,
    crp: float | None = 6.0,
    pat_global: float | None = 20.0,
    ph_global: float | None = 20.0,
) -> dict[str, object]:
    return {
        "pat_ID": patient_id,
        "Visit_months_from_diagnosis": float(month),
        "Age_diagnosis": 54,
        "Sex": 1,
        "RF_positivity": 1,
        "anti_CCP": 1,
        "DAS28": das28,
        "CRP": crp,
        "Pat_global": pat_global,
        "Ph_global": ph_global,
        "Pain": 40.0,
        "HAQ": 1.2,
        "GC": 0,
        "GC_type": 1,
        "csDMARD1": 1,
        "csDMARD2": None,
        "csDMARD3": None,
        "N_prev_csDMARD": 1,
        "bDMARD": bdmard,
        "N_prev_bDMARD": n_prev_bdmard,
        "tsDMARD": tsdmard,
        "N_prev_tsDMARD": n_prev_tsdmard,
        "Year_diagnosis": year_diagnosis,
        "month_diagnosis": 1,
    }


def test_selected_definition_uses_current_plus_prior_visit_only() -> None:
    rows = [
        _visit("P1", 0.0, das28=2.0, pat_global=60.0, bdmard=1),
        _visit("P1", 6.0, das28=3.0, pat_global=60.0, bdmard=2, tsdmard=1),
        _visit("P1", 12.0, das28=3.6, pat_global=60.0, bdmard=2, tsdmard=1),
        _visit("P1", 24.0, das28=3.4, pat_global=60.0, bdmard=2, tsdmard=1),
    ]

    visits = derive_d2t_ra_visit_flags(pd.DataFrame(rows))
    twelve_month = visits.loc[visits["Visit_months_from_diagnosis"] == 12.0].iloc[0]
    legacy_three_visit_average = (2.0 + 3.0 + 3.6) / 3.0

    assert twelve_month["rolling_avg_DAS28"] == pytest.approx((3.0 + 3.6) / 2.0)
    assert legacy_three_visit_average < 3.2
    assert bool(twelve_month["D2T_crit1"]) is True
    assert bool(twelve_month["D2T_crit2"]) is True
    assert bool(twelve_month["D2T_RA"]) is True


def test_selected_definition_crp_threshold_is_strictly_greater_than_10_mg_l() -> None:
    rows = [
        _visit("P10", 0.0, crp=9.0, pat_global=60.0, bdmard=1),
        _visit("P10", 6.0, crp=10.0, pat_global=60.0, bdmard=2, tsdmard=1),
        _visit("P10", 12.0, crp=10.0, pat_global=60.0, bdmard=2, tsdmard=1),
        _visit("P10", 24.0, crp=10.0, pat_global=60.0, bdmard=2, tsdmard=1),
        _visit("P11", 0.0, crp=9.0, pat_global=60.0, bdmard=1),
        _visit("P11", 6.0, crp=10.2, pat_global=60.0, bdmard=2, tsdmard=1),
        _visit("P11", 12.0, crp=10.2, pat_global=60.0, bdmard=2, tsdmard=1),
        _visit("P11", 24.0, crp=10.2, pat_global=60.0, bdmard=2, tsdmard=1),
    ]

    visits = derive_d2t_ra_visit_flags(pd.DataFrame(rows))
    grouped = visits.groupby("pat_ID")["D2T_crit2"].max().to_dict()

    assert grouped == {"P10": False, "P11": True}


def test_incident_preprocessing_excludes_patients_already_on_third_bts_moa_at_baseline() -> None:
    rows = [
        _visit("PX", 0.0, n_prev_bdmard=2, bdmard=3, pat_global=60.0, das28=3.6),
        _visit("PX", 6.0, n_prev_bdmard=2, bdmard=3, pat_global=60.0, das28=3.6),
        _visit("PX", 24.0, n_prev_bdmard=2, bdmard=3, pat_global=60.0, das28=3.6),
        _visit("PC", 0.0, bdmard=1, das28=2.4, crp=6.0, pat_global=20.0),
        _visit("PC", 6.0, bdmard=2, tsdmard=1, das28=2.8, crp=7.0, pat_global=20.0),
        _visit("PC", 24.0, bdmard=2, tsdmard=1, das28=2.7, crp=7.5, pat_global=20.0),
    ]

    km = strata_fit_data_to_km_input(pd.DataFrame(rows))

    assert km["pat_ID"].tolist() == ["PC"]
    assert km.iloc[0]["event_type"] == "censored"
    assert km.iloc[0]["interval_start"] == pytest.approx(24.0)


def test_raw_cox_preprocessing_uses_selected_definition_and_censoring() -> None:
    rows = [
        _visit("PE", 0.0, das28=2.0, pat_global=60.0, bdmard=1),
        _visit("PE", 6.0, das28=3.4, pat_global=60.0, bdmard=2, tsdmard=1),
        _visit("PE", 12.0, das28=3.5, pat_global=60.0, bdmard=2, tsdmard=1),
        _visit("PE", 24.0, das28=3.5, pat_global=60.0, bdmard=2, tsdmard=1),
        _visit("PC", 0.0, das28=2.1, pat_global=20.0, bdmard=1),
        _visit("PC", 6.0, das28=2.4, pat_global=20.0, bdmard=2, tsdmard=1),
        _visit("PC", 24.0, das28=2.5, pat_global=20.0, bdmard=2, tsdmard=1),
    ]

    cox = strata_fit_data_to_cox_input(
        pd.DataFrame(rows),
        time_col="time",
        outcome_col="event",
        expl_vars=["DAS28", "CRP"],
    )
    observed = dict(zip(cox["pat_ID"], cox["event"], strict=True))
    times = dict(zip(cox["pat_ID"], cox["time"], strict=True))

    assert observed == {"PE": 1, "PC": 0}
    assert times["PE"] == pytest.approx(12.0)
    assert times["PC"] == pytest.approx(24.0)


def test_latest_prevalence_year_is_marked_partial_by_default() -> None:
    rows = [
        _visit("P20", 0.0, year_diagnosis=2018, pat_global=60.0, bdmard=1, das28=2.0),
        _visit("P20", 6.0, year_diagnosis=2018, pat_global=60.0, bdmard=2, tsdmard=1, das28=3.4),
        _visit("P20", 12.0, year_diagnosis=2018, pat_global=60.0, bdmard=2, tsdmard=1, das28=3.5),
        _visit("P20", 24.0, year_diagnosis=2018, pat_global=60.0, bdmard=2, tsdmard=1, das28=3.5),
        _visit("P20", 29.0, year_diagnosis=2018, pat_global=60.0, bdmard=2, tsdmard=1, das28=3.5),
        _visit("P21", 0.0, year_diagnosis=2018, pat_global=20.0, bdmard=1, das28=2.0),
        _visit("P21", 6.0, year_diagnosis=2018, pat_global=20.0, bdmard=2, tsdmard=1, das28=2.4),
        _visit("P21", 24.0, year_diagnosis=2018, pat_global=20.0, bdmard=2, tsdmard=1, das28=2.4),
        _visit("P21", 29.0, year_diagnosis=2018, pat_global=20.0, bdmard=2, tsdmard=1, das28=2.4),
    ]

    prevalence = compute_d2t_prevalence_by_year(pd.DataFrame(rows))
    latest = prevalence.sort_values("Year_visit").iloc[-1]

    assert int(latest["Year_visit"]) == 2020
    assert bool(latest["partial_year"]) is True
