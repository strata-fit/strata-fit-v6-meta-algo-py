from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .exceptions import RuntimeInputError, TaskExecutionError
from .service import run_partial_method


@dataclass
class _SubClient:
    parent: "InProcessAlgorithmClient"


@dataclass
class _OrganizationScopedClient:
    parent: "InProcessAlgorithmClient"
    organization_id: int

    def __getattr__(self, name: str) -> Any:
        return getattr(self.parent, name)


class InProcessAlgorithmClient:
    def __init__(self, datasets: list[pd.DataFrame], organization_ids: list[int] | None = None) -> None:
        if not datasets:
            raise RuntimeInputError("At least one dataset is required")
        resolved_ids = organization_ids or list(range(len(datasets)))
        if len(resolved_ids) != len(datasets):
            raise RuntimeInputError("organization_ids length must match datasets length")
        self._datasets = {
            int(org_id): dataset.copy() for org_id, dataset in zip(resolved_ids, datasets)
        }
        self._tasks: dict[int, dict[str, Any]] = {}
        self._next_task_id = 1
        self.task = _TaskClient(self)
        self.result = _ResultClient(self)
        self.organization = _OrganizationClient(self)

    def _new_task_id(self) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def wait_for_results(self, task_id: int, interval: float = 1.0) -> list[Any]:
        del interval
        task = self._tasks.get(task_id)
        if task is None:
            raise TaskExecutionError(f"Unknown task id: {task_id}")
        return list(task["results"])


class _TaskClient(_SubClient):
    def create(
        self,
        *,
        input_: dict[str, Any],
        organizations: list[int] | None = None,
        name: str = "subtask",
        description: str | None = None,
    ) -> dict[str, Any]:
        del name, description
        organizations = organizations or []
        method = input_.get("method")
        kwargs = input_.get("kwargs", {})
        if not isinstance(method, str) or not isinstance(kwargs, dict):
            raise RuntimeInputError("Task input requires string 'method' and dict 'kwargs'")

        task_id = self.parent._new_task_id()
        if method == "main":
            from .central import main

            result = main(client=self.parent, output_path=None, **kwargs)
            results = [result]
        else:
            results = []
            for org_id in organizations:
                if org_id not in self.parent._datasets:
                    raise TaskExecutionError(f"Unknown organization id: {org_id}")
                results.append(
                    run_partial_method(
                        method,
                        df=self.parent._datasets[org_id],
                        raw_input=kwargs,
                        client=_OrganizationScopedClient(self.parent, int(org_id)),
                    )
                )

        self.parent._tasks[task_id] = {
            "input_": input_,
            "organizations": list(organizations),
            "results": results,
        }
        return {"id": task_id}


class _ResultClient(_SubClient):
    def get(self, task_id: int) -> Any:
        task = self.parent._tasks.get(task_id)
        if task is None:
            raise TaskExecutionError(f"Unknown task id: {task_id}")
        results = task["results"]
        if len(results) == 1:
            return results[0]
        return list(results)


class _OrganizationClient(_SubClient):
    def list(self) -> list[dict[str, int]]:
        return [{"id": org_id} for org_id in self.parent._datasets]


def run_local_main(
    datasets: list[pd.DataFrame],
    *,
    organization_ids: list[int] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    from .central import main

    client = InProcessAlgorithmClient(datasets, organization_ids=organization_ids)
    return main(client=client, output_path=None, **kwargs)
