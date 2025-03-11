import importlib
from enum import Enum
from functools import lru_cache
from typing import Dict, Any, Callable
from ....config.backend import Backend

class ModelName(Enum):
    LEE = "lee"

@lru_cache(maxsize=None)
def import_module_by_backend(backend: Backend, model_name: ModelName):
    try:
        module_path = f"..backend.{backend.value}.{model_name.value}_{backend.value}"
        return importlib.import_module(module_path, package = __package__)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Could not import {model_name.value} for backend {backend}: {e}")

class FunctionRegistry:
    def __init__(self):
        self._registry: Dict[Backend, Dict[str, Any]] = {backend: {} for backend in Backend}
        self._required_functions = [
            "compute_rrsw",
            "compute_rrsb",
            "compute_rrsm",
            "compute_rrsp"
        ]

    def load_functions(self, backend: Backend):
        module = import_module_by_backend(backend, ModelName.LEE)
        
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

FUNCTION_REGISTRY_LEE = {
    "rrsw": {"model": create_model_function("compute_rrsw")},
    "rrsb": {"model": create_model_function("compute_rrsb")},
    "rrsm": {"model": create_model_function("compute_rrsm")},
    "rrsp": {"model": create_model_function("compute_rrsp")},
}