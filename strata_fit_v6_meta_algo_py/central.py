from strata_fit_v6_imputation_py import STRATEGY, imputation_factory
from .utils import run_task

@algorithm_client
def main_logic(imputation_config: dict, lr_config: dict):
    imputation_model = imputation_factory(imputation_config)
    result = run_task(imputation_model.central, imputation_config)
    imputation_model = postprocess_imputation_result(result)
    imputation_params_cache = imputation_model.cache()

    # add cached imputation params to the lr config
    lr_config.append(imputation_params_cache)
    lr_model = lr_factory(lr_config)

    for i in range(lr_congig.iterations):
        result = run_task(
            impute_and_lr_learning_partial,
            dict(
                imputation_model_config=imputation_config,
                lr_model_config=lr_config,
            )
        )
        lr_model.aggregate(result["weights"])
        lr_params_cache = lr_model.cache()
        lr_config.append(lr_params_cache)

    return lr_params_cache, imputation_params_cache



