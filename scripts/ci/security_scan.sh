#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SECURITY_DIR="${SECURITY_ARTIFACT_DIR:-$ROOT_DIR/ARTIFACTS/security}"
ENV_DIR="${V6_SECURITY_ENV_DIR:-/tmp/strata-meta-security-${RUN_ID}}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-${PYTHON_BIN:-python3}}"
LANE_FILE="$SECURITY_DIR/security_lane.json"
REPORT_FILE="$SECURITY_DIR/report.json"

mkdir -p "$SECURITY_DIR"
start_epoch="$(date +%s)"
status="fail"
error_summary="security scan failed"

cleanup() {
  local end_epoch duration
  end_epoch="$(date +%s)"
  duration="$((end_epoch - start_epoch))"
  cat > "$LANE_FILE" <<EOF
{
  "lane_name": "security",
  "status": "$status",
  "duration_s": $duration,
  "report": "$REPORT_FILE",
  "error_summary": "$error_summary"
}
EOF
}
trap cleanup EXIT
trap 'error_summary="security scan failed"' ERR

rm -rf "$ENV_DIR"
"$PYTHON_BOOTSTRAP" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

# Match the runtime image surface, not the dev/infra-client environment.
"$ENV_DIR/bin/python" -m pip install --no-deps -e "$ROOT_DIR"
"$ENV_DIR/bin/python" -m pip install --no-cache-dir \
  dynaconf \
  numpy \
  pandas \
  pydantic \
  PyJWT \
  requests \
  scikit-learn \
  scipy
"$ENV_DIR/bin/python" -m pip install --no-cache-dir --no-deps \
  "strata-fit-v6-data-validator-py @ https://github.com/strata-fit/strata-fit-data-schema/archive/c77d319b6539bdc48314738981b5bde478d2bacd.tar.gz"
"$ENV_DIR/bin/python" -m pip install --no-cache-dir --no-deps \
  "v6-federated-algo-core-py @ https://github.com/mdw-nl/v6-federated-algo-core-v6/archive/c29dd63f40c6e3997a0865cb0cbc81dd9ce02a60.tar.gz"

"$ENV_DIR/bin/python" -m pip freeze > "$SECURITY_DIR/pip_freeze.txt"

if ! "$ENV_DIR/bin/python" -m pip_audit --version >/dev/null 2>&1; then
  "$ENV_DIR/bin/python" -m pip install --no-cache-dir pip-audit >/dev/null 2>&1 || true
fi

if "$ENV_DIR/bin/python" -m pip_audit --version >/dev/null 2>&1; then
  set +e
  "$ENV_DIR/bin/python" -m pip_audit --format json --output "$SECURITY_DIR/pip_audit.json"
  echo "$?" > "$SECURITY_DIR/pip_audit.exit_code"
  set -e
else
  echo '{"dependencies":[],"skipped":"pip-audit unavailable"}' > "$SECURITY_DIR/pip_audit.json"
  echo "127" > "$SECURITY_DIR/pip_audit.exit_code"
fi

if ! "$ENV_DIR/bin/python" -m bandit --version >/dev/null 2>&1; then
  "$ENV_DIR/bin/python" -m pip install --no-cache-dir bandit >/dev/null 2>&1 || true
fi

if "$ENV_DIR/bin/python" -m bandit --version >/dev/null 2>&1; then
  set +e
  "$ENV_DIR/bin/python" -m bandit -r "$ROOT_DIR/strata_fit_v6_meta_algo_py" -x "$ROOT_DIR/tests" -f json -o "$SECURITY_DIR/bandit.json" -q
  echo "$?" > "$SECURITY_DIR/bandit.exit_code"
  set -e
else
  echo '{"results":[],"skipped":"bandit unavailable"}' > "$SECURITY_DIR/bandit.json"
  echo "127" > "$SECURITY_DIR/bandit.exit_code"
fi

if command -v trivy >/dev/null 2>&1 && [ -n "${V6_SECURITY_IMAGE:-}" ]; then
  set +e
  trivy image --format json --severity HIGH,CRITICAL --ignore-unfixed --output "$SECURITY_DIR/trivy.json" "$V6_SECURITY_IMAGE"
  echo "$?" > "$SECURITY_DIR/trivy.exit_code"
  set -e
else
  echo '{"Results":[],"skipped":"trivy unavailable or V6_SECURITY_IMAGE unset"}' > "$SECURITY_DIR/trivy.json"
  echo "127" > "$SECURITY_DIR/trivy.exit_code"
fi

"$ENV_DIR/bin/python" - "$ROOT_DIR" "$SECURITY_DIR" "$REPORT_FILE" <<'PY'
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
security_dir = Path(sys.argv[2])
report_file = Path(sys.argv[3])

static_targets = [
    root / "Dockerfile",
    root / "pyproject.toml",
    root / "scripts" / "ci" / "bootstrap_meta_env.sh",
]
forbidden = {
    "harbor": re.compile(r"harbor2?\.vantage6\.ai", re.I),
    "vantage6_algorithm_tools": re.compile(r"vantage6-algorithm-tools", re.I),
    "web_server_runtime": re.compile(r"\b(fastapi|uvicorn|gunicorn|python-multipart)\b", re.I),
    "runtime_git_apt": re.compile(r"\bapt(-get)?\s+install\b.*\bgit\b", re.I),
    "mutable_git_ref": re.compile(r"@(main|master|develop|dev)(?:[\"'#\s]|$)", re.I),
}
static_findings = []
for path in static_targets:
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    for name, pattern in forbidden.items():
        for match in pattern.finditer(text):
            static_findings.append(
                {
                    "rule": name,
                    "path": str(path.relative_to(root)),
                    "match": match.group(0),
                }
            )

pip_audit = json.loads((security_dir / "pip_audit.json").read_text(encoding="utf-8"))
pip_audit_exit = (security_dir / "pip_audit.exit_code").read_text(encoding="utf-8").strip()
pip_dependencies = pip_audit.get("dependencies", []) if isinstance(pip_audit, dict) else []
pip_vulns = []
for dep in pip_dependencies:
    for vuln in dep.get("vulns", []):
        pip_vulns.append(
            {
                "name": dep.get("name"),
                "id": vuln.get("id"),
                "aliases": vuln.get("aliases", []),
                "fix_versions": vuln.get("fix_versions", []),
                "severity": vuln.get("severity"),
            }
        )

bandit = json.loads((security_dir / "bandit.json").read_text(encoding="utf-8"))
bandit_exit = (security_dir / "bandit.exit_code").read_text(encoding="utf-8").strip()
bandit_results = bandit.get("results", []) if isinstance(bandit, dict) else []
bandit_high = [
    item
    for item in bandit_results
    if item.get("issue_severity") == "HIGH" and item.get("issue_confidence") in {"MEDIUM", "HIGH"}
]

trivy = json.loads((security_dir / "trivy.json").read_text(encoding="utf-8"))
trivy_exit = (security_dir / "trivy.exit_code").read_text(encoding="utf-8").strip()
trivy_results = trivy.get("Results", []) if isinstance(trivy, dict) else []
trivy_vulns = []
for result in trivy_results:
    for vuln in result.get("Vulnerabilities", []) or []:
        trivy_vulns.append(
            {
                "target": result.get("Target"),
                "id": vuln.get("VulnerabilityID"),
                "package": vuln.get("PkgName"),
                "severity": vuln.get("Severity"),
                "fixed_version": vuln.get("FixedVersion"),
            }
        )
trivy_blockers = [
    item
    for item in trivy_vulns
    if item.get("severity") in {"HIGH", "CRITICAL"} and item.get("fixed_version")
]

blockers = []
blockers.extend({"source": "static", **item} for item in static_findings)
blockers.extend({"source": "bandit", **item} for item in bandit_high)
blockers.extend({"source": "trivy", **item} for item in trivy_blockers)

report = {
    "run_id": None,
    "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "status": "pass" if not blockers else "fail",
    "policy": {
        "fail_on": [
            "static surface regressions",
            "bandit HIGH severity with MEDIUM/HIGH confidence",
            "trivy HIGH/CRITICAL image findings with fixed versions",
        ],
        "warn_on": [
            "pip-audit findings without severity",
            "medium/low findings",
            "unavailable optional tools",
            "unfixed image vulnerabilities",
        ],
    },
    "static_findings": static_findings,
    "skipped_tools": [
        name
        for name, exit_code in {
            "pip-audit": pip_audit_exit,
            "bandit": bandit_exit,
            "trivy": trivy_exit,
        }.items()
        if exit_code == "127"
    ],
    "pip_audit": {
        "vulnerability_count": len(pip_vulns),
        "fixable_count": sum(1 for item in pip_vulns if item.get("fix_versions")),
        "findings": pip_vulns,
    },
    "bandit": {
        "finding_count": len(bandit_results),
        "high_confident_count": len(bandit_high),
    },
    "trivy": {
        "finding_count": len(trivy_vulns),
        "fixable_high_or_critical_count": len(trivy_blockers),
    },
    "blockers": blockers,
}
report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(report["status"])
raise SystemExit(0 if report["status"] == "pass" else 1)
PY

status="pass"
error_summary=""
