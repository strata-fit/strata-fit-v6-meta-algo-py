from __future__ import annotations

import json

from pathlib import Path

import pandas as pd

from strata_fit_v6_meta_algo_py.central import main
from strata_fit_v6_meta_algo_py.local_client import InProcessAlgorithmClient
from strata_fit_v6_meta_algo_py.partial import d2t_characteristics, imputation_compute_partial
from strata_fit_v6_meta_algo_py.runtime import RunContext


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pat_ID": range(1, 8),
            "time": [float(x + 1) for x in range(7)],
            "event": [1 if x % 2 == 0 else 0 for x in range(7)],
            "x1": [float(x + 1) for x in range(7)],
            "x2": [float(x + 2) for x in range(7)],
            "outcome": [1 if x % 3 == 0 else 0 for x in range(7)],
        }
    )


def test_run_context_partial_writes_output(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_path = tmp_path / "out.json"
    _dataset().to_csv(dataset_path, index=False)

    context = RunContext(
        source=tmp_path / "run_context.json",
        payload={
            "entrypoint": {"name": "imputation_compute_partial"},
            "arguments": {
                "named": {"columns": ["x1", "x2"], "imputation_strategy": "mean"}
            },
            "inputs": [{"uri": str(dataset_path)}],
            "outputs": [{"uri": str(output_path)}],
        },
    )

    result = imputation_compute_partial(run_context=context)
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == result


def test_d2t_characteristics_run_context_writes_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_path = tmp_path / "out.json"
    _dataset().to_csv(dataset_path, index=False)
    calls = []

    def fake_d2t_characteristics_frame(df, **kwargs):
        calls.append((df, kwargs))
        return {"d2t_patients": 3, "age_count": 3}

    monkeypatch.setattr(
        "strata_fit_v6_meta_algo_py.partial.d2t_characteristics_frame",
        fake_d2t_characteristics_frame,
    )

    context = RunContext(
        source=tmp_path / "run_context.json",
        payload={
            "entrypoint": {"name": "d2t_characteristics"},
            "arguments": {
                "named": {
                    "global_metrics": {},
                    "imputation_strategy": "mean",
                    "cohort": {},
                    "event_definition": "d2t_ra_v2026_selected_v1",
                }
            },
            "inputs": [{"uri": str(dataset_path)}],
            "outputs": [{"uri": str(output_path)}],
        },
    )

    result = d2t_characteristics(run_context=context)

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    assert result == {"d2t_patients": 3, "age_count": 3}
    assert len(calls) == 1
    assert calls[0][1]["global_metrics"] == {}
    assert calls[0][1]["event_definition"] == "d2t_ra_v2026_selected_v1"


def test_in_process_client_drives_central_flow(tmp_path: Path) -> None:
    datasets = [_dataset(), _dataset(), _dataset()]
    output_path = tmp_path / "central.json"
    client = InProcessAlgorithmClient(datasets)

    result = main(
        client=client,
        output_path=output_path,
        columns=["x1", "x2"],
        run_validation=False,
        imputation_strategy="mean",
        final_model="sklearn_linear",
        final_model_config={
            "predictors": ["x1", "x2"],
            "outcome": "outcome",
            "n_local_iterations": 8,
        },
    )

    assert output_path.exists()
    assert result["final_model"] == "sklearn_linear"
    assert result["final_result"]["partials"]
