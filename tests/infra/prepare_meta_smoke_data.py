#!/usr/bin/env python3
"""Generate synthetic STRATA-FIT CSV buckets for meta-algo infra smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tests.synthetic_strata_fit_data import SyntheticConfig, write_partitioned_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--node-count", type=int, default=3)
    parser.add_argument("--patients-per-node", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20260318)
    args = parser.parse_args()

    paths = write_partitioned_csv(
        output_dir=Path(args.output_dir),
        config=SyntheticConfig(
            node_count=args.node_count,
            patients_per_node=args.patients_per_node,
            seed=args.seed,
        ),
    )

    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
