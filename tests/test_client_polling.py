from __future__ import annotations

import itertools

import pytest

from strata_fit_v6_meta_algo_py import client as client_module
from strata_fit_v6_meta_algo_py.client import (
    AlgorithmProxyClient,
    AlgorithmProxyRequestError,
    _TaskClient,
)


def _client_with_statuses(statuses: list[object]) -> AlgorithmProxyClient:
    client = AlgorithmProxyClient.__new__(AlgorithmProxyClient)
    iterator = iter(statuses)

    def request(endpoint: str, **kwargs):
        del endpoint, kwargs
        next_value = next(iterator)
        if isinstance(next_value, Exception):
            raise next_value
        return next_value

    client.request = request
    return client


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = itertools.count()
    monkeypatch.setattr(client_module.time, "time", lambda: float(next(ticks)))
    monkeypatch.setattr(client_module.time, "sleep", lambda _seconds: None)


def test_wait_for_task_completion_recovers_from_transient_proxy_error(fake_clock) -> None:
    client = _client_with_statuses(
        [
            AlgorithmProxyRequestError(
                "task/1437/status",
                500,
                "Request failed, see node logs",
            ),
            {"status": "running"},
            {"status": "completed"},
        ]
    )

    client.wait_for_task_completion(1437, interval=0.1)


def test_wait_for_task_completion_fails_after_transient_error_budget(fake_clock) -> None:
    client = _client_with_statuses(
        [
            AlgorithmProxyRequestError("task/1437/status", 500, "temporary proxy failure"),
            AlgorithmProxyRequestError("task/1437/status", 500, "temporary proxy failure"),
            AlgorithmProxyRequestError("task/1437/status", 500, "temporary proxy failure"),
        ]
    )

    with pytest.raises(RuntimeError, match="Task 1437 status polling failed"):
        client.wait_for_task_completion(
            1437,
            interval=0.1,
            max_poll_error_seconds=2,
        )


def test_wait_for_task_completion_does_not_retry_client_errors(fake_clock) -> None:
    client = _client_with_statuses(
        [AlgorithmProxyRequestError("task/1437/status", 404, "not found")]
    )

    with pytest.raises(AlgorithmProxyRequestError, match="status 404"):
        client.wait_for_task_completion(1437, interval=0.1)


def test_wait_for_task_completion_fails_on_terminal_child_failure(fake_clock) -> None:
    client = _client_with_statuses([{"status": "crashed"}])

    with pytest.raises(RuntimeError, match="Task 1437 finished with status 'crashed'"):
        client.wait_for_task_completion(1437, interval=0.1)


def test_request_with_transient_retries_recovers(fake_clock) -> None:
    client = _client_with_statuses(
        [
            AlgorithmProxyRequestError("result", 500, "temporary proxy failure"),
            {"data": []},
        ]
    )

    assert client._request_with_transient_retries("result") == {"data": []}


def test_task_create_retries_transient_proxy_error(fake_clock) -> None:
    client = AlgorithmProxyClient.__new__(AlgorithmProxyClient)
    client.image = "example/image:latest"
    client.collaboration_id = 1
    client.databases = []
    client.node_id = 10
    client.study_id = None
    client.store_id = None
    calls: list[tuple[str, str]] = []
    responses = iter(
        [
            AlgorithmProxyRequestError("task", 500, "temporary proxy failure"),
            {"id": 123},
        ]
    )

    def request(endpoint: str, **kwargs):
        calls.append((endpoint, kwargs.get("method", "get")))
        next_value = next(responses)
        if isinstance(next_value, Exception):
            raise next_value
        return next_value

    client.request = request
    client.task = _TaskClient(client)

    result = client.task.create(input_={"method": "partial"}, organizations=[6])

    assert result == {"id": 123}
    assert calls == [("task", "post"), ("task", "post")]
