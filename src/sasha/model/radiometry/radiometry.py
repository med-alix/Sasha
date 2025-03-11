from functools import partial
from pydantic import Field, PrivateAttr
from ...model.design.design_base import Design
from ...model.lee.lee_model import LeeModel
from ...model.common.core import SpectralComponentLoader
from ...config.backend import Backend, convert_array
from ...model.radiometry.radiometry_registry import get_integrate_reflectance
from typing import Any




class Radiometry(SpectralComponentLoader):
    wvl: Any = Field(..., description="Array of wavelengths for the radiometry data.")
    design: Design
    reflectance_model: LeeModel
    prop_type: str
    _reflectance: Any = PrivateAttr()
    _response_matrix: Any = PrivateAttr()
    _integrate_reflectance: Any = PrivateAttr()

    def __init__(self, **data):
        if 'wvl' not in data and 'design' in data:
            data['wvl'] = data['design'].wvl
        super().__init__(**data)
        self._initialize_components()

    def _initialize_components(self):
        self.reflectance_model.wvl  = self.wvl
        self._reflectance           = partial(self.reflectance_model.get_property, prop_type=self.prop_type)
        self._response_matrix       = convert_array(self.design.response.response_matrix, self.backend)
        self._integrate_reflectance = get_integrate_reflectance(self.backend)

    def update_backend(self, new_backend: Backend):
        self.backend = new_backend
        self._initialize_components()

    def sampled_reflectance(self, **parameters):
        if self.design.model_option == "dirac":
            return self._reflectance(**parameters)
        else:
            reflectance = self._reflectance(**parameters)
            return self._integrate_reflectance(reflectance, self._response_matrix)



# Usage example:
# radiometry = Radiometry(wvl=some_wavelengths, design=design, reflectance_model=lee_model, prop_type="rrsb", backend=Backend.JAX)
# result = radiometry.sampled_reflectance(**some_parameters)