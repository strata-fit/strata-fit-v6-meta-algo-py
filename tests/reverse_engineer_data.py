from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def make_patient_rows(
    patient_id: str,
    rng: np.random.Generator,
    cohort: str,
    n_visits: int,
) -> list[dict]:
    cohort = cohort.lower()

    severity_shift = {"alpha": -0.3, "beta": 0.0, "gamma": 0.4}.get(cohort, 0.0)
    treatment_shift = {"alpha": -0.15, "beta": 0.0, "gamma": 0.2}.get(cohort, 0.0)
    d2t_bias = {"alpha": 0.15, "beta": 0.30, "gamma": 0.50}.get(cohort, 0.25)

    age = int(rng.integers(25, 80))
    sex = int(rng.integers(0, 2))
    rf = int(rng.random() < (0.55 + 0.10 * max(severity_shift, 0)))
    anti_ccp = int(rng.random() < (0.50 + 0.12 * max(severity_shift, 0)))

    baseline_das28 = clamp(rng.normal(3.8 + severity_shift, 0.5), 1.5, 7.0)
    baseline_crp = clamp(rng.normal(8 + 4 * severity_shift, 3), 0.2, 40)
    baseline_haq = clamp(rng.normal(1.0 + 0.25 * severity_shift, 0.25), 0, 3)
    baseline_pat = clamp(rng.normal(40 + 10 * severity_shift, 12), 0, 100)
    baseline_pain = clamp(rng.normal(42 + 12 * severity_shift, 14), 0, 100)

    is_future_d2t = rng.random() < d2t_bias

    months = np.cumsum(rng.integers(3, 9, size=n_visits)).astype(float)

    rows = []
    prev_cs = 0
    prev_b = 0
    prev_ts = 0

    for i, month in enumerate(months):
        progress = i / max(n_visits - 1, 1)

        worsening = (0.8 if is_future_d2t else 0.2) * progress + severity_shift * 0.2

        das28 = clamp(baseline_das28 + rng.normal(0.15 * i + worsening, 0.25), 1.2, 7.8)
        crp = clamp(baseline_crp + rng.normal(1.5 * i + 4 * worsening, 2.0), 0.1, 60)
        haq = clamp(baseline_haq + rng.normal(0.08 * i + 0.15 * worsening, 0.08), 0, 3)
        pat_global = clamp(baseline_pat + rng.normal(4 * i + 12 * worsening, 8), 0, 100)
        pain = clamp(baseline_pain + rng.normal(5 * i + 13 * worsening, 9), 0, 100)
        ph_global = clamp(pat_global - rng.normal(5, 6), 0, 100)
        esr = int(round(clamp(rng.normal(20 + 3 * i + 8 * worsening, 6), 1, 100)))
        sjc28 = int(round(clamp(rng.normal(3 + 0.7 * i + 1.5 * worsening, 2), 0, 28)))
        tjc28 = int(round(clamp(rng.normal(4 + 0.8 * i + 1.7 * worsening, 2), 0, 28)))
        eq5d = round(clamp(rng.normal(0.78 - 0.04 * i - 0.08 * worsening, 0.05), 0.05, 1.0), 3)

        treatment_pressure = progress + treatment_shift + (0.25 if is_future_d2t else 0.0)

        cs1 = 1
        cs2 = int(treatment_pressure > 0.30)
        cs3 = int(treatment_pressure > 0.55)
        bdmard = int(treatment_pressure > 0.45)
        tsdmard = int(treatment_pressure > 0.75)

        if i > 0:
            prev_cs = max(prev_cs, cs2 + cs3)
            prev_b = max(prev_b, int(rows[-1]["bDMARD"] == 1))
            prev_ts = max(prev_ts, int(rows[-1]["tsDMARD"] == 1))

        gc = int(rng.random() < (0.25 + 0.20 * worsening + 0.10 * progress))
        gc_dose = round(clamp(rng.normal(4 + 3 * worsening, 2), 0, 25), 1)

        rows.append(
            {
                "pat_ID": patient_id,
                "Visit_months_from_diagnosis": round(float(month), 1),
                "Age_diagnosis": age,
                "Sex": sex,
                "RF_positivity": rf,
                "anti_CCP": anti_ccp,
                "DAS28": round(das28, 2),
                "CRP": round(crp, 2),
                "ESR": esr,
                "HAQ": round(haq, 2),
                "Pat_global": round(pat_global, 1),
                "Pain": round(pain, 1),
                "Ph_global": round(ph_global, 1),
                "SJC28": sjc28,
                "TJC28": tjc28,
                "eq5d": eq5d,
                "csDMARD1": cs1,
                "csDMARD2": cs2,
                "csDMARD3": cs3,
                "N_prev_csDMARD": prev_cs,
                "bDMARD": bdmard,
                "N_prev_bDMARD": prev_b,
                "tsDMARD": tsdmard,
                "N_prev_tsDMARD": prev_ts,
                "GC": gc,
                "GC_dose": gc_dose,
                "D2T_RA": is_future_d2t if i == len(months) - 1 else False,
            }
        )

    return rows


def generate_cohort(cohort: str, n_patients: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    prefix = cohort[:2].upper()

    for i in range(1, n_patients + 1):
        patient_id = f"{prefix}{i:04d}"
        n_visits = int(rng.integers(4, 9))
        rows.extend(make_patient_rows(patient_id, rng, cohort, n_visits))

    df = pd.DataFrame(rows)
    df = df.sort_values(["pat_ID", "Visit_months_from_diagnosis"]).reset_index(drop=True)
    return df


def build_dataset(output_csv: Path, cohort_name: str, n_patients: int, seed: int) -> None:
    df = generate_cohort(cohort_name, n_patients=n_patients, seed=seed)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} rows to {output_csv}")


if __name__ == "__main__":
    base = Path("v6-infra/infrastructure/data/meta")

    build_dataset(base / "alpha_generated.csv", "alpha", n_patients=60, seed=101)
    build_dataset(base / "beta_generated.csv", "beta", n_patients=60, seed=202)
    build_dataset(base / "gamma_generated.csv", "gamma", n_patients=60, seed=303)