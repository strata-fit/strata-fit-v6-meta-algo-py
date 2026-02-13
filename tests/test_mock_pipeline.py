from unittest import result
import pandas as pd
from pydantic import BaseModel
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

from strata_fit_v6_meta_algo_py.central import run_pipeline
from strata_fit_v6_meta_algo_py.partial import impute_locally

from strata_fit_v6_imputation_py.imputation_strategies.mean import MeanImputer
from v6_logistic_regression_py.partials import _logistic_regression_partial
from strata_fit_v6_data_validator_py.logic import validate_csv


def test_imputation_mean_roundtrip():
    df = pd.DataFrame(
        {
            "pat_ID": [1, 1, 2],
            "x": [1.0, None, 3.0],
        }
    )
    imputer = MeanImputer()
    metrics = imputer.compute(df, ["x"]).to_dict()
    global_metrics = imputer.aggregate([metrics], ["x"])
    imputed = imputer.impute(df, global_metrics)

    assert imputed["x"].isna().sum() == 0
    # Missing values should match the aggregated global metric
    expected = list(global_metrics["x"].values())[0]
    assert imputed.loc[df["x"].isna(), "x"].iloc[0] == expected


def test_logistic_partial_training():
    df = pd.DataFrame(
        {
            "f1": [0, 1, 0, 1],
            "f2": [1, 1, 0, 0],
            "y": [0, 1, 0, 1],
        }
    )
    init_attrs = {
        "coef_": [[0.0, 0.0]],
        "intercept_": [0.0],
        "classes_": [0, 1],
    }

    result = _logistic_regression_partial(
        df,
        model_attributes=init_attrs,
        predictors=["f1", "f2"],
        outcome="y",
        n_local_iterations=50,
    )

    assert "model_attributes" in result
    assert "coef_" in result["model_attributes"]
    assert result["size"] == 4

    coef = np.array(result["model_attributes"]["coef_"][0], dtype=float)
    intercept = float(result["model_attributes"]["intercept_"][0])

    X = df[["f1", "f2"]].to_numpy(dtype=float)
    probs = sigmoid(intercept + X @ coef)

    assert len(probs) == len(df)
    assert (probs >= 0).all() and (probs <= 1).all()

def test_validator_simple_model():
    class RowModel(BaseModel):
        age: int
        sex: str

    df = pd.DataFrame({"age": [10, "bad"], "sex": ["M", "F"]})
    has_errors, errors = validate_csv(df, RowModel)
    assert has_errors is True
    assert len(errors) == 1
    assert errors[0].field == "age"


def test_meta_run_pipeline():
    df = pd.DataFrame(
        {
            "pat_ID": [1, 1, 2, 2],
            "x": [1.0, None, 3.0, 5.0],
            "y": [0, 1, 0, 1],
        }
    )
    assert "km_pooled" in result
    assert "pooled_km_event_table" in result["km_pooled"]
    assert len(result["km_pooled"]["pooled_km_event_table"]) > 0
    assert "imputation_metrics" in result
    assert "lr_model_attributes" in result
    # ensure imputation then training used all rows
    assert result["training_size"] == 4

print("imputation_metrics:", result["imputation_metrics"])
print("lr_model_attributes:", result["lr_model_attributes"])
print("km rows:", len(result["km_pooled"]["pooled_km_event_table"]))



tbl = pd.DataFrame(result["km_pooled"]["pooled_km_event_table"])
tbl = tbl.sort_values("interval_start")

n = tbl["at_risk"].astype(float).to_numpy()
d = tbl["observed"].astype(float).to_numpy()

haz = np.where(n > 0, d / n, 0.0)
S = np.cumprod(1 - haz)

km_curve = pd.DataFrame({"t": tbl["interval_start"], "S": S})
print(km_curve.head(20))
