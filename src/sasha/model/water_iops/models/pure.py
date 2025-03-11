from typing import  Optional, Any
from pydantic import Field, PrivateAttr
from ....model.common.core import SpectralComponentLoader
from ....config.backend import Backend, convert_array
import importlib



class WaterPure(SpectralComponentLoader):
    awater: Optional[Any] = Field(default=None)
    _water_module: Any = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_properties()

    def _initialize_properties(self):
        self.awater = convert_array(self.initialize_property(self.pure_water['awater_file']),self.backend)
        self._water_module = self._get_water_module()
        self.logger.info(f"Initialized Pure Water Model")

    @property
    def bwater(self):
        Abwater = 0.00288
        Ebwater = -4.32
        return self._water_module.compute_bwater(self.wvl, Abwater, Ebwater)

    def _get_water_module(self):
        backend_mapping = {
            Backend.NUMPY: "..backend.numpy.pure_numpy",
            Backend.NUMBA: "..backend.numba.pure_numba",
            Backend.JAX: "..backend.jax.pure_jax",
            Backend.TENSORFLOW: "..backend.tensorflow.pure_tensorflow"
        }
        
        module_path = backend_mapping.get(self.backend)
        if module_path is None:
            raise ValueError(f"Invalid backend option: {self.backend}")
        
        # Add the package parameter to specify the reference point for relative import
        return importlib.import_module(module_path, package=__package__)

        
    def get_bwater(self):
        return self.bwater

# Usage example:
# water_pure = WaterPure(wvl=some_wavelengths, pure_water={'awater_file': 'path/to/file'}, backend=Backend.NUMPY)
# bwater = water_pure.get_bwater()