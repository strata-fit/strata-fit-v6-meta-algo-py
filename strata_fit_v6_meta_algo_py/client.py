from __future__ import annotations

import base64
import json
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jwt
import requests

from .log import error, info
from .runtime import get_env_var


def _json_to_b64(payload: Any) -> str:
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")


def _b64_to_json(payload: str) -> Any:
    return json.loads(base64.b64decode(payload.encode("utf-8")).decode("utf-8"))


def _has_task_finished(status: str | None) -> bool:
    return status in {
        "completed",
        "failed",
        "start failed",
        "non-existing Docker image",
        "crashed",
        "killed by user",
        "not allowed",
        "unknown error",
    }


@dataclass
class _SubClient:
    parent: "AlgorithmProxyClient"


class AlgorithmProxyClient:
    def __init__(self, token: str, host: str, port: int, path: str = "/api") -> None:
        self.token = token
        self.host = host
        self.port = port
        self.path = path

        jwt_payload = jwt.decode(token, options={"verify_signature": False})
        identity = jwt_payload.get("sub") or {}

        self.image = identity.get("image")
        self.databases = identity.get("databases", [])
        self.node_id = identity.get("node_id")
        self.collaboration_id = identity.get("collaboration_id")
        self.study_id = identity.get("study_id")
        self.store_id = identity.get("store_id")
        self.organization_id = identity.get("organization_id")

        self.task = _TaskClient(self)
        self.result = _ResultClient(self)
        self.organization = _OrganizationClient(self)

    @classmethod
    def from_env(cls) -> "AlgorithmProxyClient":
        host = get_env_var("HOST")
        port = get_env_var("PORT")
        api_path = get_env_var("API_PATH", "/api")
        token_file = get_env_var("TOKEN_FILE")

        if not host or not port or not token_file:
            raise RuntimeError("Missing one or more required Vantage6 env vars")

        token = Path(token_file).read_text(encoding="utf-8").strip()
        normalized_path = "/api" if api_path is None else api_path
        return cls(token=token, host=host, port=int(port), path=normalized_path)

    @property
    def base_url(self) -> str:
        return f"{self.host}:{self.port}{self.path}"

    def request(
        self,
        endpoint: str,
        *,
        method: str = "get",
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = requests.request(
            method.upper(),
            url,
            json=json_body,
            params=params,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=300,
        )
        if response.status_code >= 400:
            try:
                data = response.json()
            except Exception:
                data = {"msg": response.text}
            msg = data.get("msg") or response.text
            raise RuntimeError(
                f"Request to '{endpoint}' failed with status {response.status_code}: {msg}"
            )
        return response.json()

    def _multi_page_request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = dict(params or {})
        page = 1
        response = self.request(endpoint, params={**params, "page": page})
        data = response["data"]
        links = response.get("links")
        while links and links.get("next"):
            page += 1
            response = self.request(endpoint, params={**params, "page": page})
            data += response["data"]
            links = response.get("links")
        return data

    def wait_for_task_completion(self, task_id: int, interval: float = 1.0) -> None:
        started = time.time()
        while True:
            status_response = self.request(f"task/{task_id}/status")
            status = status_response.get("status")
            if _has_task_finished(status):
                elapsed = int(time.time() - started)
                info(f"Task {task_id} finished with status '{status}' after {elapsed}s")
                return
            time.sleep(interval)
            interval = min(interval * 1.5, 60.0)

    def wait_for_results(self, task_id: int, interval: float = 1.0) -> list[Any]:
        self.wait_for_task_completion(task_id, interval=interval)
        return self.result.from_task(task_id)


class _TaskClient(_SubClient):
    def create(
        self,
        *,
        input_: dict[str, Any],
        organizations: list[int] | None = None,
        name: str = "subtask",
        description: str | None = None,
    ) -> dict[str, Any]:
        organizations = organizations or []
        description = description or f"task from node_id={self.parent.node_id}"
        serialized_input = _json_to_b64(input_)
        organization_json_list = [
            {"id": org_id, "input": serialized_input} for org_id in organizations
        ]

        body: dict[str, Any] = {
            "name": name,
            "image": self.parent.image,
            "collaboration_id": self.parent.collaboration_id,
            "description": description,
            "organizations": organization_json_list,
            "databases": self.parent.databases,
        }
        if self.parent.study_id:
            body["study_id"] = self.parent.study_id
        if self.parent.store_id:
            body["store_id"] = self.parent.store_id
        return self.parent.request("task", method="post", json_body=body)


class _ResultClient(_SubClient):
    def from_task(self, task_id: int) -> list[Any]:
        raw_results = self.parent._multi_page_request("result", params={"task_id": task_id})
        decoded_results = []
        for run in raw_results:
            payload = run.get("result")
            if not payload:
                continue
            try:
                decoded_results.append(_b64_to_json(payload))
            except Exception as exc:
                error(f"Unable to decode result for task {task_id}: {exc}")
        return decoded_results


class _OrganizationClient(_SubClient):
    def get(self, organization_id: int) -> dict[str, Any]:
        return self.parent.request(f"organization/{organization_id}")

    def list(self) -> list[dict[str, Any]]:
        params: dict[str, Any]
        if self.parent.study_id:
            params = {"study_id": self.parent.study_id}
        else:
            params = {"collaboration_id": self.parent.collaboration_id}
        return self.parent._multi_page_request("organization", params=params)
