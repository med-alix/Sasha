# File: shallow_sa/model/water_iops/chl_registry.py

import importlib
from enum import Enum
from functools import lru_cache
from typing import Dict, Any, Callable
from ....config.backend import Backend

class ModelName(Enum):
    CHL = "chl"

@lru_cache(maxsize=None)
def import_module_by_backend(backend: Backend, model_name: ModelName):
    try:
        module_path = f"....model.water_iops.backend.{backend.value}.{model_name.value}_{backend.value}"
        return importlib.import_module(module_path, package = __package__)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Could not import {model_name.value} for backend {backend}: {e}")

class FunctionRegistry:
    def __init__(self):
        self._registry: Dict[Backend, Dict[str, Any]] = {backend: {} for backend in Backend}
        self._required_functions = [
            "bricaud_model",
            "log_bricaud_model",
            "two_peaks_model",
            "micro_nano_model"
        ]

    def load_functions(self, backend: Backend):
        module = import_module_by_backend(backend, ModelName.CHL)
        
        for func_name in self._required_functions:
            if not hasattr(module, func_name):
                raise AttributeError(f"{func_name} not found in {module.__name__}")
            self._registry[backend][func_name] = getattr(module, func_name)

    def get_function(self, func_name: str, backend: Backend):
        if backend not in self._registry or func_name not in self._registry[backend]:
            self.load_functions(backend)
        return self._registry[backend][func_name]

_function_registry = FunctionRegistry()

def create_model_function(func_name: str) -> Callable:
    def model_function(*args, **kwargs):
        backend = kwargs.pop("backend", Backend.NUMPY)
        return _function_registry.get_function(func_name, backend)(*args, **kwargs)
    return model_function

FUNCTION_REGISTRY_APHY = {
    "bricaud": {"model": create_model_function("bricaud_model")},
    "log_bricaud": {"model": create_model_function("log_bricaud_model")},
    "two_peaks": {"model": create_model_function("two_peaks_model")},
    "micro_nano": {"model": create_model_function("micro_nano_model")},
    "None": {"model": None},
}

FUNCTION_REGISTRY_BPHY = {"None": {"model": None}}

# Usage:
# from shallow_sa.model.water_iops.chl_registry import FUNCTION_REGISTRY_APHY, FUNCTION_REGISTRY_BPHY