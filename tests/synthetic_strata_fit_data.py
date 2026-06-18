"""Synthetic STRATA-FIT dataset generation for mock and infra E2E tests.

Generation flow:
1. Build latent one-row-per-patient Cox-style population (time/event + risk covariates).
2. Reverse-engineer longitudinal STRATA-FIT rows that satisfy PatientData schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import itertools

import numpy as np
import pandas as pd

PATIENT_DATA_COLUMNS = [
    "pat_ID",
    "Visit_months_from_diagnosis",
    "Age_diagnosis",
    "Sex",
    "RF_positivity",
    "anti_CCP",
    "DAS28",
    "Pat_global",
    "Pain",
    "Ph_global",
    "CRP",
    "ESR",
    "SJC28",
    "TJC28",
    "csDMARD1",
    "csDMARD2",
    "csDMARD3",
    "conc_MTX_dose",
    "N_prev_csDMARD",
    "bDMARD",
    "N_prev_bDMARD",
    "tsDMARD",
    "N_prev_tsDMARD",
    "GC",
    "GC_type",
    "GC_dose",
    "eq5d",
    "HAQ",
    "Year_diagnosis",
    "month_diagnosis",
    "Symptom_duration",
]


@dataclass(frozen=True)
class SyntheticConfig:
    node_count: int = 3
    patients_per_node: int = 36
    seed: int = 20260318
    missingness_profile: str = "baseline"
    event_profile: str = "adequate"
    signal_profile: str = "strong"
    site_heterogeneity: bool = False


def _build_latent_cox_population(
    total_patients: int,
    seed: int,
    *,
    event_profile: str,
    signal_profile: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(22, 82, size=total_patients)
    sex = rng.integers(0, 2, size=total_patients)
    rf = rng.integers(0, 2, size=total_patients)
    anti_ccp = rng.integers(0, 2, size=total_patients)

    das28_base = np.clip(rng.normal(loc=3.2, scale=0.7, size=total_patients), 1.2, 6.8)
    crp_base = np.clip(rng.gamma(shape=2.2, scale=2.3, size=total_patients), 0.2, 30.0)
    haq_base = np.clip(rng.normal(loc=1.0, scale=0.45, size=total_patients), 0.0, 3.0)

    if signal_profile == "null":
        linear_risk = rng.normal(0.0, 0.25, size=total_patients)
    elif signal_profile == "modest":
        linear_risk = (
            0.020 * (age - 50)
            + 0.45 * (das28_base - 3.2)
            + 0.04 * (crp_base - 5.0)
            + 0.35 * (haq_base - 1.0)
            + 0.15 * rf
            + rng.normal(0.0, 0.55, size=total_patients)
        )
    else:
        linear_risk = (
            0.040 * (age - 50)
            + 0.85 * (das28_base - 3.2)
            + 0.09 * (crp_base - 5.0)
            + 0.80 * (haq_base - 1.0)
            + 0.35 * rf
            + rng.normal(0.0, 0.45, size=total_patients)
        )

    event_shift = {
        "low": -2.25,
        "high_censoring": -1.45,
        "adequate": -0.2,
    }.get(event_profile, -0.2)
    event_prob = 1.0 / (1.0 + np.exp(-(linear_risk + event_shift)))
    event_flag = (rng.uniform(0.0, 1.0, size=total_patients) < event_prob).astype(int)

    event_time = np.where(
        event_flag == 1,
        np.clip(rng.weibull(1.6, size=total_patients) * 24 + 12, 12, 96),
        np.clip(rng.weibull(1.8, size=total_patients) * 30 + 18, 18, 108),
    )

    return pd.DataFrame(
        {
            "patient_index": np.arange(1, total_patients + 1),
            "Age_diagnosis": age,
            "Sex": sex,
            "RF_positivity": rf,
            "anti_CCP": anti_ccp,
            "DAS28_base": das28_base,
            "CRP_base": crp_base,
            "HAQ_base": haq_base,
            "cox_time": event_time,
            "cox_event": event_flag,
        }
    )


def _visit_schedule(target_time: float, event: int) -> list[float]:
    if event:
        switch = max(6.0, target_time - 6.0)
        times = [0.0, 3.0, 6.0, switch, target_time, min(target_time + 6.0, target_time + 18.0)]
    else:
        times = [0.0, 6.0, 12.0, target_time]
    deduped = sorted({round(t, 2) for t in times if t >= 0.0})
    if len(deduped) < 2:
        deduped.append(round(target_time, 2))
    return deduped


def _reverse_engineer_visits(
    latent: pd.DataFrame,
    seed: int,
    *,
    missingness_profile: str,
    site_shift: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 101)
    rows: list[dict[str, float | int | str | None]] = []

    for row in latent.itertuples(index=False):
        patient_number = int(row.patient_index)
        pat_id = f"SE{patient_number}"

        year_diag = int(rng.integers(2008, 2022))
        month_diag = int(rng.integers(1, 12))
        visit_times = _visit_schedule(float(row.cox_time), int(row.cox_event))
        dmard_switch_month = max(6.0, float(row.cox_time) - 6.0)

        for month in visit_times:
            event_visit = int(row.cox_event) == 1 and month >= float(row.cox_time)
            post_switch = int(row.cox_event) == 1 and month >= dmard_switch_month

            das28 = float(np.clip(row.DAS28_base + site_shift + (1.25 if event_visit else -0.25) + rng.normal(0, 0.2), 1.0, 7.0))
            crp = float(np.clip(row.CRP_base + site_shift * 2.5 + (2.8 if event_visit else -0.6) + rng.normal(0, 0.6), 0.1, 40.0))
            haq = float(np.clip(row.HAQ_base + site_shift * 0.2 + (0.45 if event_visit else -0.1) + rng.normal(0, 0.1), 0.0, 3.0))

            pat_global = float(np.clip(65 + rng.normal(0, 6) if event_visit else 40 + rng.normal(0, 8), 0, 100))
            ph_global = float(np.clip(62 + rng.normal(0, 7) if event_visit else 38 + rng.normal(0, 7), 0, 100))
            pain = float(np.clip((pat_global + ph_global) / 2 + rng.normal(0, 6), 0, 100))

            missingness = {
                "baseline": {"DAS28": 0.16, "CRP": 0.14, "HAQ": 0.12, "PRO": 0.00},
                "high": {"DAS28": 0.34, "CRP": 0.30, "HAQ": 0.28, "PRO": 0.18},
                "lab": {"DAS28": 0.16, "CRP": 0.42, "HAQ": 0.12, "PRO": 0.00},
            }.get(missingness_profile, {"DAS28": 0.16, "CRP": 0.14, "HAQ": 0.12, "PRO": 0.00})

            # Missingness exercises imputation without breaking D2T-defining visits.
            if not event_visit and rng.random() < missingness["DAS28"]:
                das28 = np.nan
            if not event_visit and rng.random() < missingness["CRP"]:
                crp = np.nan
            if not event_visit and rng.random() < missingness["HAQ"]:
                haq = np.nan
            if not event_visit and rng.random() < missingness["PRO"]:
                pat_global = np.nan
            if not event_visit and rng.random() < missingness["PRO"]:
                ph_global = np.nan
            if not event_visit and rng.random() < missingness["PRO"]:
                pain = np.nan

            bdmard = int(2 if post_switch else 1)
            tsdmard = int(1) if post_switch and rng.random() < 0.85 else None

            schema_row = {
                "pat_ID": pat_id,
                "Visit_months_from_diagnosis": float(month),
                "Age_diagnosis": int(row.Age_diagnosis),
                "Sex": int(row.Sex),
                "RF_positivity": int(row.RF_positivity),
                "anti_CCP": int(row.anti_CCP),
                "DAS28": das28,
                "Pat_global": pat_global,
                "Pain": pain,
                "Ph_global": ph_global,
                "CRP": crp,
                "ESR": int(np.clip(round(15 + 3.8 * np.nan_to_num(das28, nan=3.0) + rng.normal(0, 4)), 0, 120)),
                "SJC28": int(np.clip(round(2 + max(np.nan_to_num(das28, nan=3.0) - 2.2, 0) * 2.0 + rng.normal(0, 1.2)), 0, 28)),
                "TJC28": int(np.clip(round(3 + max(np.nan_to_num(das28, nan=3.0) - 2.2, 0) * 2.4 + rng.normal(0, 1.4)), 0, 28)),
                "csDMARD1": 1,
                "csDMARD2": None,
                "csDMARD3": None,
                "conc_MTX_dose": float(np.clip(15.0 + rng.normal(0, 2), 5.0, 30.0)),
                "N_prev_csDMARD": 1,
                "bDMARD": bdmard,
                "N_prev_bDMARD": int(1 if post_switch else 0),
                "tsDMARD": tsdmard,
                "N_prev_tsDMARD": int(1 if tsdmard else 0),
                "GC": int(rng.integers(0, 2)),
                "GC_type": int(rng.integers(1, 5)),
                "GC_dose": float(np.clip(5 + rng.normal(0, 1.8), 0, 12)),
                "eq5d": float(np.clip(0.85 - 0.08 * np.nan_to_num(haq, nan=1.0) + rng.normal(0, 0.04), 0, 1)),
                "HAQ": haq,
                "Year_diagnosis": year_diag,
                "month_diagnosis": month_diag,
                "Symptom_duration": float(np.clip(8 + rng.normal(0, 3), 0, 48)),
            }
            rows.append(schema_row)

    frame = pd.DataFrame(rows)
    return frame[PATIENT_DATA_COLUMNS].sort_values(["pat_ID", "Visit_months_from_diagnosis"]).reset_index(drop=True)


def generate_partitioned_strata_fit_datasets(config: SyntheticConfig) -> list[pd.DataFrame]:
    if config.node_count < 1 or config.node_count > 8:
        raise ValueError("node_count must be between 1 and 8")
    if config.patients_per_node < 12:
        raise ValueError("patients_per_node must be >= 12 for stable Cox threshold checks")

    total_patients = config.node_count * config.patients_per_node
    latent = _build_latent_cox_population(
        total_patients=total_patients,
        seed=config.seed,
        event_profile=config.event_profile,
        signal_profile=config.signal_profile,
    )

    partitions: list[pd.DataFrame] = []
    for node_index in range(config.node_count):
        start = node_index * config.patients_per_node
        stop = start + config.patients_per_node
        node_latent = latent.iloc[start:stop].copy()
        site_shift = 0.0
        if config.site_heterogeneity:
            midpoint = (config.node_count - 1) / 2
            site_shift = (node_index - midpoint) * 0.22
        node_rows = _reverse_engineer_visits(
            node_latent,
            seed=config.seed + node_index * 17,
            missingness_profile=config.missingness_profile,
            site_shift=site_shift,
        )
        partitions.append(node_rows)

    return partitions


def write_partitioned_csv(
    *,
    output_dir: Path,
    config: SyntheticConfig,
    names: Iterable[str] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = generate_partitioned_strata_fit_datasets(config)
    file_names = list(names) if names else [f"data_bucket{i + 1}.csv" for i in range(len(frames))]

    paths: list[Path] = []
    for frame, name in zip(frames, file_names):
        path = output_dir / name
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def generate_d2t_truth_table_dataset() -> pd.DataFrame:
    rows: list[dict[str, float | int | str | None]] = []
    for patient_number, (crit1, crit2, crit3) in enumerate(
        itertools.product([False, True], repeat=3),
        start=1,
    ):
        pat_id = f"D2T{patient_number}"
        for month in [0.0, 6.0, 12.0]:
            treatment_changed = crit1 and month >= 6.0
            row = {
                "pat_ID": pat_id,
                "Visit_months_from_diagnosis": month,
                "Age_diagnosis": 45 + patient_number,
                "Sex": patient_number % 2,
                "RF_positivity": int(patient_number % 2 == 0),
                "anti_CCP": int(patient_number % 3 == 0),
                "DAS28": 3.21 if crit2 else 3.2,
                "Pat_global": 51.0 if crit3 else 50.0,
                "Pain": 55.0 if crit3 else 45.0,
                "Ph_global": 40.0,
                "CRP": 1.0,
                "ESR": 18,
                "SJC28": 3,
                "TJC28": 4,
                "csDMARD1": 1,
                "csDMARD2": None,
                "csDMARD3": None,
                "conc_MTX_dose": 15.0,
                "N_prev_csDMARD": 1,
                "bDMARD": 2 if treatment_changed else 1,
                "N_prev_bDMARD": int(treatment_changed),
                "tsDMARD": None,
                "N_prev_tsDMARD": 0,
                "GC": 0,
                "GC_type": 1,
                "GC_dose": 0.0,
                "eq5d": 0.75,
                "HAQ": 1.0,
                "Year_diagnosis": 2012,
                "month_diagnosis": 1,
                "Symptom_duration": 6.0,
            }
            rows.append(row)
    return pd.DataFrame(rows)[PATIENT_DATA_COLUMNS]
