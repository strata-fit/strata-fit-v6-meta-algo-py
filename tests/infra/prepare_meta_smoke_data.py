#!/usr/bin/env python3
"""Generate synthetic CSV buckets for meta-algo infra smoke testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_bucket(bucket_index: int, rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + bucket_index)

    age = rng.integers(28, 78, size=rows).astype(float)
    das28 = np.clip(rng.normal(loc=3.8 + 0.05 * bucket_index, scale=0.7, size=rows), 1.2, 7.0)
    crp = np.clip(rng.gamma(shape=2.2, scale=2.0, size=rows), 0.1, 25.0)
    haq = np.clip(rng.normal(loc=1.0 + 0.03 * bucket_index, scale=0.35, size=rows), 0.0, 3.0)

    # Inject missing values to exercise the imputation stage.
    das28[::9] = np.nan
    crp[::7] = np.nan
    haq[::11] = np.nan

    x1 = rng.normal(loc=0.2 * bucket_index, scale=1.1, size=rows)
    x2 = rng.normal(loc=-0.1 * bucket_index, scale=0.9, size=rows)
    x1[::8] = np.nan

    # Guarantee enough observed events per node for the Cox min threshold (>=10).
    event = (np.arange(rows) % 2 == 0).astype(int)
    time = np.clip(rng.weibull(a=1.6, size=rows) * 15.0 + 1.0, 1.0, None)

    linear_signal = (
        np.nan_to_num(das28, nan=3.5) * 0.5
        + np.nan_to_num(crp, nan=4.0) * 0.12
        + np.nan_to_num(haq, nan=1.0) * 0.9
        + age * 0.01
        + rng.normal(0.0, 0.4, size=rows)
    )
    threshold = float(np.median(linear_signal))
    rf_positivity = (linear_signal >= threshold).astype(int)

    return pd.DataFrame(
        {
            "pat_ID": [f"N{bucket_index}_R{idx}" for idx in range(rows)],
            "Age_diagnosis": age,
            "DAS28": das28,
            "CRP": crp,
            "HAQ": haq,
            "RF_positivity": rf_positivity,
            "time": time,
            "event": event,
            "x1": x1,
            "x2": x2,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--node-count", type=int, default=3)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260313)
    args = parser.parse_args()

    if args.node_count < 2 or args.node_count > 8:
        raise ValueError("--node-count must be between 2 and 8")
    if args.rows < 24:
        raise ValueError("--rows must be >= 24 to satisfy Cox event threshold robustly")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bucket in range(1, args.node_count + 1):
        df = build_bucket(bucket_index=bucket, rows=args.rows, seed=args.seed)
        out_path = out_dir / f"data_bucket{bucket}.csv"
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path} rows={len(df)} events={int(df['event'].sum())}")


if __name__ == "__main__":
    main()
