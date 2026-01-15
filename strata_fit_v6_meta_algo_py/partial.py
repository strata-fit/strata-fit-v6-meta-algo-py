# TODO: call partial imputation for this local data
# Then cache it within the local LR partial function
# This might not work, so we will have to write the file on disk and use `database.type==folder`
# In that case you need to implement a temorary file that will be cleaed up (another method)
from strata_fit_v6_lr_py import lr_factory
from strata_fit_v6_imputation_py import imputation_factory


@data(1)
def impute_and_lr_learning_partial(data: pd.DataFrame, imputation_model_config: Dict[str, Any], lr_model_config: Dict[str, Any]) -> Dict[str, Any]:
    # infer the local imputation model
    imputation_model = imputation_factory(imputation_model_config)
    data = imputation_model.infer(data)

    # learn the local LR model
    lr_model = lr_factory(lr_model_config, weights=weights)
    lr_model.partial(data)

    # cache the model
    lr_params_cache = lr_model.cache()

    # return the model
    return {
        "lr_params_cache": lr_params_cache,
    }
