#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "lane_name": path.stem,
            "status": "fail",
            "duration_s": 0,
            "error_summary": f"missing artifact: {path}",
        }
    return json.loads(path.read_text())


def maybe_git_rev() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR).decode().strip()
        return out
    except Exception:
        return "unknown"


def maybe_python_version() -> str:
    return sys.version.split()[0]


def main() -> int:
    run_id = os.getenv("RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    artifact_dir = Path(os.getenv("ARTIFACT_DIR", str(ROOT_DIR / "ARTIFACTS" / "validation")))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    lanes = [
        read_json(artifact_dir / "clean_env_lane.json"),
        read_json(artifact_dir / "stress_matrix_lane.json"),
        read_json(artifact_dir / "infra_lane.json"),
    ]
    statuses = [lane.get("status", "fail") for lane in lanes]
    overall_status = "pass" if all(item == "pass" for item in statuses) else "fail"

    report = {
        "run_id": run_id,
        "timestamp_utc": now_utc(),
        "git_rev": maybe_git_rev(),
        "environment": {
            "python_version": maybe_python_version(),
            "cwd": str(ROOT_DIR),
        },
        "lanes": lanes,
        "overall_status": overall_status,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2))

    lines = [
        f"# Validation Report ({run_id})",
        "",
        f"- Timestamp (UTC): {report['timestamp_utc']}",
        f"- Git revision: {report['git_rev']}",
        f"- Overall status: **{overall_status.upper()}**",
        "",
        "## Lanes",
    ]
    for lane in lanes:
        lines.append(
            f"- `{lane.get('lane_name', 'unknown')}`: {lane.get('status', 'fail')} ({lane.get('duration_s', 0)}s)"
        )
        if lane.get("error_summary"):
            lines.append(f"  - error: {lane['error_summary']}")

    (artifact_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"overall_status={overall_status}")
    return 0 if overall_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
