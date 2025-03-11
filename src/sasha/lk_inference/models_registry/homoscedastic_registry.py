import importlib
from ...config.backend import Backend



def get_homo_var_module(backend):
    if backend == Backend.JAX :
        return  importlib.import_module(
        "..backend.jax.homoscedastic_jax", package = __package__
    )
    elif backend == Backend.TENSORFLOW:
        return  importlib.import_module(
        "..backend.tensorflow.homoscedastic_tensor", package = __package__
    )
    elif backend == Backend.NUMPY:
        return  importlib.import_module(
        "..backend.numpy.homoscedastic_numpy",  package = __package__
    )
    else:
        raise ValueError("Invalid backend option")

def get_functions_registry(backend):
    homo_var_module  = get_homo_var_module(backend)
    log_likelihood   = homo_var_module.log_likelihood
    forward_model    = homo_var_module.forward_model
    model_jacob      = homo_var_module.model_jacob_numerical
    generate_samples = homo_var_module.generate_samples
    score = homo_var_module.score_numerical
    efim = homo_var_module.efim_numerical
    ofim = homo_var_module.ofim_numerical

    FUNCTION_HOMOSCEDASTIC_REGISTRY = {
    "log_likelihood": log_likelihood,
    "forward_model": forward_model,
    "model_jacob": model_jacob,
    "score": score,
    "efim": efim,
    "ofim": ofim,
    "generate_samples": generate_samples,
    }
    return FUNCTION_HOMOSCEDASTIC_REGISTRY