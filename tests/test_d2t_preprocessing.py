from __future__ import annotations

import itertools

import pandas as pd

from strata_fit_v6_meta_algo_py.preprocessing import derive_d2t_ra_visit_flags, strata_fit_data_to_cox_input


def _patient_rows(
    patient_id: str,
    *,
    crit1: bool,
    crit2: bool,
    crit3: bool,
    das28: float | None = None,
    crp: float | None = None,
    pat_global: float | None = None,
    ph_global: float | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for month in [0.0, 6.0, 12.0]:
        treatment_changed = crit1 and month >= 6.0
        rows.append(
            {
                "pat_ID": patient_id,
                "Visit_months_from_diagnosis": month,
                "DAS28": das28 if das28 is not None else (3.21 if crit2 else 3.2),
                "CRP": crp if crp is not None else (1.01 if crit2 else 1.0),
                "Pat_global": pat_global if pat_global is not None else (51.0 if crit3 else 50.0),
                "Ph_global": ph_global if ph_global is not None else (40.0 if crit3 else 50.0),
                "bDMARD": 2 if treatment_changed else 1,
                "tsDMARD": None,
                "Year_diagnosis": 2012,
            }
        )
    return rows


def test_d2t_requires_all_three_criteria() -> None:
    rows: list[dict[str, object]] = []
    expected: dict[str, bool] = {}
    for index, (crit1, crit2, crit3) in enumerate(itertools.product([False, True], repeat=3), start=1):
        patient_id = f"P{index}"
        rows.extend(_patient_rows(patient_id, crit1=crit1, crit2=crit2, crit3=crit3))
        expected[patient_id] = crit1 and crit2 and crit3

    visits = derive_d2t_ra_visit_flags(pd.DataFrame(rows))
    observed = visits.groupby("pat_ID")["D2T_RA"].max().to_dict()

    assert observed == expected
    assert (visits["D2T_RA"] == (visits["D2T_crit1"] & visits["D2T_crit2"] & visits["D2T_crit3"])).all()


def test_d2t_threshold_boundaries_are_intentional() -> None:
    boundary_false = pd.DataFrame(
        _patient_rows(
            "P1",
            crit1=True,
            crit2=False,
            crit3=False,
            das28=3.2,
            crp=1.0,
            pat_global=50.0,
            ph_global=50.0,
        )
    )
    visits = derive_d2t_ra_visit_flags(boundary_false)
    last = visits.sort_values("Visit_months_from_diagnosis").iloc[-1]
    assert bool(last["D2T_crit1"]) is True
    assert bool(last["D2T_crit2"]) is False
    assert bool(last["D2T_crit3"]) is False
    assert bool(last["D2T_RA"]) is False

    boundary_true = pd.DataFrame(
        _patient_rows(
            "P2",
            crit1=True,
            crit2=True,
            crit3=True,
            das28=3.21,
            crp=1.0,
            pat_global=50.01,
            ph_global=50.0,
        )
    )
    visits = derive_d2t_ra_visit_flags(boundary_true)
    last = visits.sort_values("Visit_months_from_diagnosis").iloc[-1]
    assert bool(last["cum_unique_btsDMARD"] >= 2) is True
    assert bool(last["months_since_last_dmard"] >= 6) is True
    assert bool(last["D2T_crit2"]) is True
    assert bool(last["D2T_crit3"]) is True
    assert bool(last["D2T_RA"]) is True


def test_raw_cox_preprocessing_uses_d2t_events_only_when_all_criteria_hold() -> None:
    rows = []
    rows.extend(_patient_rows("P1", crit1=True, crit2=True, crit3=True))
    rows.extend(_patient_rows("P2", crit1=True, crit2=True, crit3=False))
    rows.extend(_patient_rows("P3", crit1=False, crit2=True, crit3=True))

    cox = strata_fit_data_to_cox_input(
        pd.DataFrame(rows),
        time_col="time",
        outcome_col="event",
        expl_vars=["DAS28", "CRP"],
    )
    observed = dict(zip(cox["pat_ID"], cox["event"], strict=True))
    assert observed == {"P1": 1, "P2": 0, "P3": 0}
