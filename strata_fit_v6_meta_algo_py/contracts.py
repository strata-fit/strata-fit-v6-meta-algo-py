from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, RootModel

from strata_fit_v6_imputation_py.imputation_strategies.base import ImputationStrategyEnum


class FinalModelEnum(str, Enum):
    SKLEARN_LINEAR = "sklearn_linear"
    COX = "cox"
    KM = "km"


class SklearnLinearFinalConfig(BaseModel):
    predictors: List[str] = Field(min_length=1)
    outcome: str
    n_local_iterations: int = 50
    model_class: Any = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)


class CoxFinalConfig(BaseModel):
    time_col: str
    outcome_col: str
    expl_vars: List[str] = Field(min_length=1)
    max_iterations: int = 10
    tolerance: float = 1e-6
    preprocess_raw_data: bool = False


class KMFinalConfig(BaseModel):
    preprocess_raw_data: bool = True


class MetaCentralInput(BaseModel):
    columns: List[str] = Field(min_length=1)
    organizations: Optional[List[int]] = None
    model_name: Optional[str] = None
    run_validation: bool = True
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER
    final_model: FinalModelEnum = FinalModelEnum.SKLEARN_LINEAR
    final_model_config: Dict[str, Any] = Field(default_factory=dict)


class MetaCentralOutput(BaseModel):
    organizations: List[int]
    validation: List[Dict[str, Any]] = Field(default_factory=list)
    imputation_metrics: Dict[str, Any]
    final_model: FinalModelEnum
    final_result: Dict[str, Any]


class ValidatePartialInput(BaseModel):
    model_name: Optional[str] = None


class ValidatePartialOutput(BaseModel):
    total_rows: int
    total_errors: int
    error_rate_per_row: float
    validation_passed: bool


class ImputationComputePartialInput(BaseModel):
    columns: List[str] = Field(min_length=1)
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER


class ImputationComputePartialOutput(RootModel[Dict[str, Any]]):
    pass


class ImputeAndTrainSklearnLinearInput(BaseModel):
    global_metrics: Dict[str, Any]
    predictors: List[str] = Field(min_length=1)
    outcome: str
    n_local_iterations: int = 50
    model_class: Any = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER


class ImputeAndTrainSklearnLinearOutput(BaseModel):
    model_attributes: Dict[str, Any]
    size: int


class CoxGetUniqueEventTimesImputedInput(BaseModel):
    time_col: str
    outcome_col: str
    global_metrics: Dict[str, Any]
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER
    minimum_events: int = 10
    preprocess_raw_data: bool = False


class CoxGetUniqueEventTimesImputedOutput(BaseModel):
    times: Optional[Dict[str, Dict[Any, Any]]] = None
    n_threshold_not_met: Optional[int] = None


class CoxComputeSummedZImputedInput(BaseModel):
    outcome_col: str
    expl_vars: List[str] = Field(min_length=1)
    global_metrics: Dict[str, Any]
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER
    preprocess_raw_data: bool = False


class CoxComputeSummedZImputedOutput(BaseModel):
    sum: Dict[str, float]


class CoxPerformIterationImputedInput(BaseModel):
    time_col: str
    expl_vars: List[str] = Field(min_length=1)
    beta: List[float]
    unique_time_events: List[float]
    global_metrics: Dict[str, Any]
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER
    preprocess_raw_data: bool = False


class CoxPerformIterationImputedOutput(BaseModel):
    agg1: List[float]
    agg2: Dict[str, Dict[Any, float]]
    agg3: List[List[List[float]]]


class KMGetUniqueEventTimesImputedInput(BaseModel):
    global_metrics: Dict[str, Any]
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER
    preprocess_raw_data: bool = True


class KMGetUniqueEventTimesImputedOutput(BaseModel):
    times: List[float] = Field(default_factory=list)


class KMGetEventTableImputedInput(BaseModel):
    unique_event_times: List[float] = Field(default_factory=list)
    global_metrics: Dict[str, Any]
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER
    preprocess_raw_data: bool = True


class KMGetEventTableImputedOutput(BaseModel):
    table: Dict[str, List[float]]
