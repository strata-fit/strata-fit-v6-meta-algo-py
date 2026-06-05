from __future__ import annotations

from typing import Any

import pandas as pd

from v6_federated_core import MethodContext, dispatch_registered_method, to_v6_result

from .methods import METHOD_REGISTRY, build_min_organization_policies, build_policy_context


def run_partial_method(
    method_name: str,
    *,
    df: pd.DataFrame,
    raw_input: dict[str, Any] | None = None,
    client: Any = None,
) -> dict[str, Any]:
    context = MethodContext(
        method=method_name,
        organization_ids=[],
        meta={"df": df, "client": client},
    )
    result = dispatch_registered_method(
        METHOD_REGISTRY,
        method_name,
        raw_input or {},
        context=context,
    )
    return to_v6_result(result)


def run_central_method(
    *,
    raw_input: dict[str, Any],
    client: Any,
    organization_ids: list[int],
) -> dict[str, Any]:
    context = MethodContext(
        method="main",
        organization_ids=organization_ids,
        meta={"client": client},
    )
    result = dispatch_registered_method(
        METHOD_REGISTRY,
        "main",
        raw_input,
        context=context,
        policies=build_min_organization_policies(),
        policy_context=build_policy_context("main", organization_ids),
    )
    return to_v6_result(result)
