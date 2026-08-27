import pandas as pd

from strata_fit_v6_meta_algo_py.contracts import SurvivalBundleFinalConfig
from strata_fit_v6_meta_algo_py.methods import _aggregate_d2t_characteristics
from strata_fit_v6_meta_algo_py.preprocessing import derive_paper_cox_covariates


def test_paper_cox_covariates_use_requested_reference_groups() -> None:
    frame = pd.DataFrame(
        {
            "Sex": [1, 0, 1, 1],
            "RF_positivity": [0, 1, 1, None],
            "anti_CCP": [0, 0, 1, 0],
            "Year_diagnosis": [2005, 2008, 2013, 2020],
            "_serology_missing_original": [False, False, False, True],
        }
    )
    result = derive_paper_cox_covariates(frame)
    assert result["Sex_Female"].tolist() == [1, 0, 1, 1]
    assert result["Serology_Either"].tolist() == [0, 1, 0, 0]
    assert result["Serology_Both"].tolist() == [0, 0, 1, 0]
    assert result["Serology_Missing"].tolist() == [0, 0, 0, 1]
    assert result["Diagnosis_year_2006_2010"].tolist() == [0, 1, 0, 0]
    assert len(SurvivalBundleFinalConfig().expl_vars) == 8


def test_characteristics_components_aggregate_mean_sd_and_percentages() -> None:
    result = _aggregate_d2t_characteristics(
        [
            {
                "d2t_patients": 2,
                "female_count": 2, "female_sum": 1, "female_sum_sq": 1,
                "rf_count": 2, "rf_sum": 1, "rf_sum_sq": 1,
                "anti_ccp_count": 2, "anti_ccp_sum": 2, "anti_ccp_sum_sq": 2,
                "age_count": 2, "age_sum": 100, "age_sum_sq": 5200,
                "das28_count": 2, "das28_sum": 8, "das28_sum_sq": 34,
            }
        ]
    )
    assert result["female_percentage"] == 50.0
    assert result["rf_positive_percentage"] == 50.0
    assert result["anti_ccp_positive_percentage"] == 100.0
    assert result["age_mean"] == 50.0
    assert result["age_sd"] == 14.142135623730951
    assert result["das28_mean_at_d2t"] == 4.0
