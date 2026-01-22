import pandas as pd
from pydantic import BaseModel

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

    result = run_pipeline(
        df,
        imputation_columns=["x"],
        predictors=["x"],
        outcome="y",
        n_local_iterations=30,
    )

    assert "imputation_metrics" in result
    assert "lr_model_attributes" in result
    # ensure imputation then training used all rows
    assert result["training_size"] == 4
