"""Mathematical audit of the federated survival pipeline.

Purpose
-------
Check, on small synthetic data, that the federated code computes what it
claims to compute. Each section below maps to a finding ID in
``FEDERATED_LOCAL_ALIGNMENT_PLAN.md`` (§2 and Appendix A).

How to read the results
-----------------------
* Tests WITHOUT a marker are correctness guarantees that hold today
  (e.g. the federated Cox arithmetic equals a pooled reference).
  These are IMPLEMENTATION checks on identical synthetic inputs, so they use
  tight tolerances. Comparison with partners' local results uses different,
  looser criteria and lives outside this file.
* Tests marked ``xfail(strict=True)`` document a KNOWN DEFECT. They assert the
  *correct* behaviour, so they fail today. When a fix lands they start
  passing, ``strict=True`` turns that into a failure, and the marker must be
  removed - so the plan and the code cannot silently drift apart.

How to run
----------
    .venv/bin/python -m pytest tests/test_math_audit.py -v     # as tests
    .venv/bin/python tests/test_math_audit.py                  # JSON report

The JSON report reproduces the numbers quoted in the plan. The report captured
at commit 7b0ccc1 is stored in ``tests/fixtures/math_audit_baseline_7b0ccc1.json``.
"""

from __future__ import annotations

import contextlib
import io
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from v6_federated_core import MethodContext

from strata_fit_v6_meta_algo_py.contracts import CoxFinalConfig
from strata_fit_v6_meta_algo_py.cox import (
    GetUniqueEventTimesInput,
    PerformIterationInput,
    compute_derivatives,
    get_unique_event_times_handler,
    perform_iteration_handler,
)
from strata_fit_v6_meta_algo_py.imputation import ImputationStrategyEnum, MeanImputer
from strata_fit_v6_meta_algo_py.local_client import InProcessAlgorithmClient
from strata_fit_v6_meta_algo_py.methods import _run_cox_with_imputation
from strata_fit_v6_meta_algo_py.preprocessing import derive_paper_cox_covariates


# =============================================================================
# Shared synthetic data and an independent reference implementation
# =============================================================================
#
# One continuous covariate x, true log-hazard ratio 0.65.
# Event times are rounded UP to whole months, so there are tied event times,
# like real STRATA-FIT visit data. Censoring is uniform on 3..24 months.

SEED = 782
N_PATIENTS = 90


def make_survival_data(seed: int = SEED, n: int = N_PATIENTS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    event_time = np.ceil(rng.exponential(12 * np.exp(-0.65 * x)))
    censor_time = rng.integers(3, 25, size=n)
    return pd.DataFrame(
        {
            "pat_ID": np.arange(n),
            "time": np.minimum(event_time, censor_time),
            "event": (event_time <= censor_time).astype(int),
            "x": x,
        }
    )


def breslow_log_likelihood(beta: np.ndarray, data: pd.DataFrame) -> float:
    """Cox partial log-likelihood with Breslow tie handling, written from the formula.

        l(beta) = sum_events x_i * beta  -  sum_t d_t * log( sum_{j: T_j >= t} exp(x_j * beta) )

    where d_t is the number of events at distinct event time t. This is
    deliberately independent of the package code so it can serve as a reference.
    """
    x = data["x"].to_numpy()
    t = data["time"].to_numpy()
    is_event = data["event"].to_numpy().astype(bool)

    total = float(x[is_event].sum() * beta[0])
    for event_time in np.unique(t[is_event]):
        d_t = np.sum(is_event & (t == event_time))
        risk_set = t >= event_time
        total -= d_t * np.log(np.exp(x[risk_set] * beta[0]).sum())
    return total


def split_into_nodes(data: pd.DataFrame, n_nodes: int = 3) -> List[pd.DataFrame]:
    """Split rows into contiguous chunks, each acting as one federated node."""
    return [data.iloc[idx] for idx in np.array_split(np.arange(len(data)), n_nodes)]


def fit_federated_cox(datasets: List[pd.DataFrame], max_iterations: int = 10) -> Dict[str, Any]:
    """Run the real federated Cox workflow in-process, one dataset per node.

    No imputation is applied (empty global metrics, all covariates complete),
    and raw-data preprocessing is off, so only the Cox arithmetic is tested.
    """
    config = CoxFinalConfig(
        time_col="time",
        outcome_col="event",
        expl_vars=["x"],
        preprocess_raw_data=False,
        max_iterations=max_iterations,
        cohort={},
    )
    with contextlib.redirect_stdout(io.StringIO()):  # silence node logging
        return _run_cox_with_imputation(
            InProcessAlgorithmClient(datasets),
            list(range(len(datasets))),
            config,
            {},
            ImputationStrategyEnum.MEAN_IMPUTER,
        )


# =============================================================================
# Section 1 - Mean imputation aggregate (findings B1, P1)
# =============================================================================
#
# Two patients, one variable x, six visits:
#   * "all missing":    patient a has x missing at all 4 visits; b has 4, 4.
#                        Observed-value mean = 4.0.
#   * "partly missing": patient a has 2 then 3 missing; b has 8, 8.
#                        Observed-value mean = (2 + 8 + 8) / 3 = 6.0.
#
# The node payload holds per-patient means plus visit counts n. The central
# step then weights each patient mean by n (ALL visits, not observed ones)
# and treats an all-missing patient's mean as 0. Both are wrong for an
# observed-value mean, and sending per-patient rows is a privacy defect.

MEAN_CASES = {
    "all_missing_patient": [np.nan, np.nan, np.nan, np.nan, 4.0, 4.0],
    "partial_missing": [2.0, np.nan, np.nan, np.nan, 8.0, 8.0],
}


def audit_mean_imputation(values: List[float]) -> Dict[str, Any]:
    df = pd.DataFrame({"pat_ID": ["a"] * 4 + ["b"] * 2, "x": values})
    imputer = MeanImputer()
    node_payload = imputer.compute(df, ["x"])  # what leaves the node
    global_metric = imputer.aggregate([node_payload], ["x"])  # central step
    return {
        "actual_mean": float(global_metric["x"][0]),
        "observed_mean": float(df["x"].mean()),  # pandas skips NaN: the correct value
        "payload_keys": list(node_payload),
    }


@pytest.mark.xfail(strict=True, reason="B1: mean aggregate weights by all visits and counts missing as 0")
@pytest.mark.parametrize("case", list(MEAN_CASES))
def test_mean_imputation_equals_observed_mean(case: str) -> None:
    result = audit_mean_imputation(MEAN_CASES[case])
    assert result["actual_mean"] == pytest.approx(result["observed_mean"])


@pytest.mark.xfail(strict=True, reason="P1: node payload contains patient IDs and per-patient rows")
def test_mean_imputation_payload_is_node_level_only() -> None:
    # Dropping the pat_ID column alone is not enough: any per-patient row
    # (e.g. one mean and visit count per patient) is still patient-level data.
    # A node-level payload holds exactly one value per variable/statistic.
    df = pd.DataFrame({"pat_ID": ["a"] * 4 + ["b"] * 2, "x": MEAN_CASES["partial_missing"]})
    payload = MeanImputer().compute(df, ["x"])
    assert "pat_ID" not in payload
    for statistic in payload.values():
        values = statistic.values() if isinstance(statistic, dict) else [statistic]
        assert len(list(values)) == 1


@pytest.mark.xfail(strict=True, reason="B1: a variable with no observed values gets mean 0 instead of NaN")
def test_mean_of_all_missing_variable_is_undefined() -> None:
    # If nothing is observed anywhere there is no mean. Returning 0.0 would
    # silently impute every value as zero (e.g. DAS28 = 0, i.e. remission).
    imputer = MeanImputer()
    payload = imputer.compute(pd.DataFrame({"pat_ID": ["a", "b"], "x": [np.nan, np.nan]}), ["x"])
    assert np.isnan(imputer.aggregate([payload], ["x"])["x"][0])


# =============================================================================
# Section 2 - Mean-imputed categorical covariates (finding E1)
# =============================================================================
#
# Mean imputation turns binary/categorical fields into fractions, e.g.
#   Sex = 2/3            -> coded Sex_Female = 0 (patient becomes "male")
#   Year_diagnosis = 2010.4 -> falls between the 2006-2010 and 2011-2015 bins,
#                              so all year indicators are 0, which is exactly
#                              the coding of the <2006 REFERENCE group.
# A fractional value must not silently map onto a valid category.

def audit_categorical_encoding() -> Dict[str, Any]:
    imputed_row = pd.DataFrame(
        {"Sex": [2 / 3], "RF_positivity": [0.0], "anti_CCP": [0.0], "Year_diagnosis": [2010.4]}
    )
    return derive_paper_cox_covariates(imputed_row).to_dict(orient="records")[0]


@pytest.mark.xfail(strict=True, reason="E1: fractional Sex is silently coded as male")
def test_fractional_sex_is_not_coded_as_male() -> None:
    assert audit_categorical_encoding()["Sex_Female"] != 0.0


@pytest.mark.xfail(strict=True, reason="E1: fractional year is silently coded as the <2006 reference group")
def test_fractional_year_is_not_coded_as_reference_group() -> None:
    row = audit_categorical_encoding()
    year_flags = [row[f"Diagnosis_year_{span}"] for span in ("2006_2010", "2011_2015", "2016_2024")]
    assert any(flag != 0.0 for flag in year_flags)


# =============================================================================
# Section 3 - Federated Cox derivatives vs numerical derivatives
# =============================================================================
#
# At a fixed beta = 0.23, each of 3 nodes computes its risk-set sums
# (agg1 = sum exp(xb), agg2 = sum x exp(xb), agg3 = sum x x' exp(xb)) for the
# global list of event times. The central step adds them up and builds the
# gradient and Hessian. Those must equal finite-difference derivatives of the
# reference Breslow log-likelihood on the POOLED data.
#
# The Hessian tolerance is looser because the numerical second derivative
# is itself a nested finite difference (error ~1e-4 here).

def audit_derivatives() -> Dict[str, float]:
    data = make_survival_data()
    beta = np.array([0.23])
    event_times = sorted(data.loc[data["event"] == 1, "time"].unique())

    # Node side: risk-set aggregates per node.
    node_payloads = []
    with contextlib.redirect_stdout(io.StringIO()):
        for node_df in split_into_nodes(data, 3):
            node_payloads.append(
                perform_iteration_handler(
                    PerformIterationInput(
                        time_col="time", expl_vars=["x"], beta=beta.tolist(), unique_time_events=event_times
                    ),
                    context=MethodContext(method="audit", meta={"df": node_df}),
                )
            )

    # Central side: sum the node aggregates, then form the derivatives.
    agg1 = sum(np.array(p["agg1"]) for p in node_payloads)
    agg2 = sum(pd.DataFrame(p["agg2"]).to_numpy() for p in node_payloads)
    agg3 = sum(np.array(p["agg3"]) for p in node_payloads)
    events_per_time = pd.DataFrame(
        {
            "time": event_times,
            "freq": [int(((data["time"] == t) & (data["event"] == 1)).sum()) for t in event_times],
        }
    )
    z_sum = np.array([data.loc[data["event"] == 1, "x"].sum()])
    gradient, hessian = compute_derivatives(agg1, agg2, agg3, events_per_time, z_sum)

    # Reference: finite differences of the pooled log-likelihood.
    def loglik(b: np.ndarray) -> np.ndarray:
        return np.array([breslow_log_likelihood(b, data)])

    gradient_ref = approx_derivative(loglik, beta).reshape(-1)
    hessian_ref = approx_derivative(lambda b: approx_derivative(loglik, b).reshape(-1), beta)
    return {
        "gradient_error": float(np.abs(gradient - gradient_ref).max()),
        "hessian_error": float(np.abs(hessian - hessian_ref).max()),
    }


def test_federated_derivatives_match_numerical_reference() -> None:
    errors = audit_derivatives()
    assert errors["gradient_error"] < 1e-6
    assert errors["hessian_error"] < 1e-2


# =============================================================================
# Section 4 - Federated Cox estimate vs pooled reference (complete data)
# =============================================================================
#
# Three estimates of the same beta:
#   reference  - scipy BFGS maximising the independent Breslow likelihood
#   pooled     - our workflow with ALL data in a single node
#   federated  - our workflow with the data split over 3 nodes
# With complete covariates and every node above the size threshold they
# must agree (the workflow rounds beta to 5 decimals, hence the tolerance).

def audit_cox_complete() -> Dict[str, Any]:
    data = make_survival_data()
    reference = minimize(lambda b: -breslow_log_likelihood(b, data), [0.0], method="BFGS", tol=1e-10)
    pooled = fit_federated_cox([data])
    federated = fit_federated_cox(split_into_nodes(data, 3))
    return {
        "reference_beta": float(reference.x[0]),
        "pooled_beta": pooled["coefficients"][0]["coef"],
        "federated_beta": federated["coefficients"][0]["coef"],
        "converged": federated["converged"],
    }


def test_federated_cox_matches_pooled_reference() -> None:
    result = audit_cox_complete()
    assert result["converged"]
    assert result["pooled_beta"] == pytest.approx(result["reference_beta"], abs=1e-4)
    assert result["federated_beta"] == pytest.approx(result["reference_beta"], abs=1e-4)


# =============================================================================
# Section 5 - Fit statistics computed at the returned beta (finding B3)
# =============================================================================
#
# Stop after ONE Newton step, i.e. a NON-converged run, so beta has clearly
# moved. Final statistics must be recomputed at the returned beta after ANY
# termination, converged or not. The reported AIC should be
#     -2 * loglik(returned beta) + 2 * p.
# Today the workflow combines the NEW beta with risk-set sums from the
# PREVIOUS beta (same for SE/Hessian), so AIC and SE describe a point that is
# not the returned estimate. Near convergence the gap shrinks, which is why
# it is easy to miss.

def audit_one_iteration() -> Dict[str, Any]:
    data = make_survival_data()
    one_step = fit_federated_cox(split_into_nodes(data, 3), max_iterations=1)
    beta = one_step["coefficients"][0]["coef"]
    n_params = 1
    return {
        "beta": beta,
        "reported_aic": one_step["aic"],
        "reference_aic_at_reported_beta": -2 * breslow_log_likelihood([beta], data) + 2 * n_params,
        "converged": one_step["converged"],
    }


@pytest.mark.xfail(strict=True, reason="B3: AIC/SE use the previous iteration's risk sets")
def test_aic_is_evaluated_at_returned_beta() -> None:
    result = audit_one_iteration()
    assert result["reported_aic"] == pytest.approx(result["reference_aic_at_reported_beta"], rel=1e-6)


# =============================================================================
# Section 6 - Consistent rows across Cox steps (finding B2)
# =============================================================================
#
# Set x to missing for ONE patient who has an event. The risk-set step
# (perform_iteration) drops rows with missing covariates, but the event-time
# step still counts that patient's event. Numerator and denominator of the
# partial likelihood then come from different populations.

def audit_missing_covariate_events() -> Dict[str, int]:
    data = make_survival_data()
    first_event_row = data.index[data["event"] == 1][0]
    data.loc[first_event_row, "x"] = np.nan

    with contextlib.redirect_stdout(io.StringIO()):
        event_table = get_unique_event_times_handler(
            GetUniqueEventTimesInput(time_col="time", outcome_col="event"),
            context=MethodContext(method="audit", meta={"df": data}),
        )
    return {
        "events_counted": int(pd.DataFrame(event_table["times"])["freq"].sum()),
        "complete_case_events": int(data.dropna(subset=["x"])["event"].sum()),
    }


@pytest.mark.xfail(strict=True, reason="B2: event counts include rows dropped from risk sets")
def test_event_count_uses_complete_cases() -> None:
    result = audit_missing_covariate_events()
    assert result["events_counted"] == result["complete_case_events"]


# =============================================================================
# Section 7 - Small-node exclusion (finding F1): arithmetic vs. policy
# =============================================================================
#
# Same 90 patients, split 9 / 81. The node with 9 patients falls under the
# "minimum_events = 10" threshold (which actually counts ROWS, not events),
# so it is excluded from Cox. The coefficient then moves from the all-patient
# value (0.808) to 0.907 purely because of how the data are distributed.
#
# Two separate questions, deliberately tested separately:
#   (a) ARITHMETIC: with the SAME included patients, federated must equal
#       pooled. Holds today.
#   (b) POLICY: who is excluded, and is it reported? Exclusion is a
#       disclosure/eligibility decision, not a maths bug, so it is not an
#       xfail here. Note that switching to an EVENT threshold would not change
#       this case (9 patients can never have 10 events), and statistically a
#       node with few or zero events can still contribute to shared-baseline
#       risk sets. The policy is tracked in the plan, not asserted here.
#
# Splits keep each patient's row intact (one row per patient in Cox input).

def audit_small_node_partition() -> Dict[str, Any]:
    data = make_survival_data()
    small_node, large_node = data.iloc[:9], data.iloc[9:]
    all_patients = fit_federated_cox([data])
    included_only = fit_federated_cox([large_node])
    split = fit_federated_cox([small_node, large_node])
    return {
        "included": split["included_organizations"],
        "excluded": split["excluded_organizations"],
        "beta": split["coefficients"][0]["coef"],
        "pooled_beta": all_patients["coefficients"][0]["coef"],
        "pooled_beta_included_patients_only": included_only["coefficients"][0]["coef"],
    }


def test_federated_equals_pooled_on_same_included_patients() -> None:
    result = audit_small_node_partition()
    assert result["beta"] == pytest.approx(result["pooled_beta_included_patients_only"], abs=1e-5)


def test_small_node_exclusion_is_reported() -> None:
    result = audit_small_node_partition()
    assert result["excluded"] == [0]
    assert result["included"] == [1]


# =============================================================================
# Section 8 - Risk-set differencing reveals individual covariates (finding P2)
# =============================================================================
#
# Each node sends, per global event time t, the risk-set sums
#     agg1(t) = sum exp(x b),   agg2(t) = sum x exp(x b)   over {T >= t}.
# Subtracting two consecutive event times leaves the sums over the patients
# who LEFT the risk set in between. If exactly one patient left, then
#     (agg2(t_k) - agg2(t_k+1)) / (agg1(t_k) - agg1(t_k+1)) = x of that patient.
# So "only aggregates leave the node" does not by itself protect individuals.
# The same holds for per-event-time Efron sums (one event: E1/E0 = x) and for
# the per-node sum of event covariates when a node has a single event.
# The test asserts the protective property, so it is xfail until mitigated
# (e.g. minimum risk-set change per step, coarser time grid, secure aggregation).

def audit_risk_set_differencing() -> Dict[str, Any]:
    data = make_survival_data()
    beta = np.array([0.23])
    event_times = sorted(data.loc[data["event"] == 1, "time"].unique())
    with contextlib.redirect_stdout(io.StringIO()):
        payload = perform_iteration_handler(
            PerformIterationInput(time_col="time", expl_vars=["x"], beta=beta.tolist(), unique_time_events=event_times),
            context=MethodContext(method="audit", meta={"df": data}),
        )
    agg1 = np.array(payload["agg1"])
    agg2 = pd.DataFrame(payload["agg2"])["x"].to_numpy()

    # Find a step where exactly one patient leaves the risk set (the attacker
    # only needs to guess this; here we pick it from the data to verify).
    for k in range(len(event_times) - 1):
        leaving = data[(data["time"] >= event_times[k]) & (data["time"] < event_times[k + 1])]
        if len(leaving) == 1:
            recovered_x = (agg2[k] - agg2[k + 1]) / (agg1[k] - agg1[k + 1])
            return {"recovered_x": float(recovered_x), "true_x": float(leaving["x"].iloc[0])}
    return {"recovered_x": None, "true_x": None}


@pytest.mark.xfail(strict=True, reason="P2: consecutive risk-set sums reveal an individual's covariate")
def test_risk_set_sums_do_not_reveal_individual_covariates() -> None:
    result = audit_risk_set_differencing()
    if result["true_x"] is None:
        pytest.skip("synthetic data has no single-patient risk-set step")
    assert result["recovered_x"] != pytest.approx(result["true_x"], abs=1e-6)


# =============================================================================
# Report: `python tests/test_math_audit.py` prints all raw numbers as JSON
# =============================================================================

def build_report() -> Dict[str, Any]:
    return {
        "all_missing_patient": audit_mean_imputation(MEAN_CASES["all_missing_patient"]),
        "partial_missing": audit_mean_imputation(MEAN_CASES["partial_missing"]),
        "categorical": [audit_categorical_encoding()],
        "derivatives": audit_derivatives(),
        "cox_complete": audit_cox_complete(),
        "one_iteration": audit_one_iteration(),
        "missing_cox": audit_missing_covariate_events(),
        "small_node_partition": audit_small_node_partition(),
        "risk_set_differencing": audit_risk_set_differencing(),
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
