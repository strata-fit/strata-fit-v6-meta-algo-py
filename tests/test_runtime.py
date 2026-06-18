from __future__ import annotations

import json

from pathlib import Path

import pytest

from strata_fit_v6_meta_algo_py.runtime import (
    RunContext,
    _resolve_entrypoint_callable,
    decode_env_value,
    main,
)


def test_run_context_helpers(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "out.json"
    context_path = tmp_path / "run_context.json"
    context_path.write_text(
        json.dumps(
            {
                "entrypoint": {"name": "main"},
                "arguments": {"named": {"columns": ["x1"]}},
                "inputs": [{"uri": str(input_path)}],
                "outputs": [{"uri": str(output_path)}],
            }
        ),
        encoding="utf-8",
    )

    context = RunContext.from_path(context_path)
    assert context.entrypoint_name() == "main"
    assert context.named_args() == {"columns": ["x1"]}
    assert context.input_uris() == [input_path]
    assert context.output_uris() == [output_path]


def test_decode_env_value_passthrough_for_plain_values() -> None:
    assert decode_env_value("/tmp/run_context.json") == "/tmp/run_context.json"


def test_entrypoint_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()

    class FakeEntryPoint:
        def __init__(self, name: str) -> None:
            self.name = name
            self.dist = type("Dist", (), {"name": "strata_fit_v6_meta_algo_py"})()

        def load(self):
            return sentinel

    monkeypatch.setattr(
        "strata_fit_v6_meta_algo_py.runtime.entry_points",
        lambda *, group: [FakeEntryPoint("main")],
    )

    assert _resolve_entrypoint_callable("main") is sentinel


def test_main_requires_run_context_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUN_CONTEXT_FILE", raising=False)
    with pytest.raises(RuntimeError, match="RUN_CONTEXT_FILE"):
        main()
