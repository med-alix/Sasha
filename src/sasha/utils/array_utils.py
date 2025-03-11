from typing import Any, Union, List
from ..config.backend import Backend, backend_manager, convert_array
import tensorflow as tf

def get_array_module():
    if backend_manager.backend == Backend.TENSORFLOW:
        import tensorflow as tf
        return tf
    elif backend_manager.backend in [Backend.NUMPY, Backend.NUMBA]:
        import numpy as np
        return np
    elif backend_manager.backend == Backend.JAX:
        import jax.numpy as jnp
        return jnp
    else:
        raise ValueError(f"Unsupported backend: {backend_manager.backend}")

def ensure_array_param_dict(kwargs: dict, backend = None) -> dict:
    filtered_dict = {}
    # keys_to_check = ["chl", "cdom", "tsm", "z"]
    for key in list(kwargs.keys()):
        # if key in keys_to_check or "alpha_m" in key:
            filtered_dict[key] = convert_array(kwargs[key], backend = backend )
    return filtered_dict


def generate_dict(arr_2d, values):
    if len(arr_2d) != len(values):
        raise ValueError("Number of spectra and values count must be the same.")
    dict_ = {}
    for i, val in enumerate(values):
        dict_[val] = arr_2d[i]
    return dict_
