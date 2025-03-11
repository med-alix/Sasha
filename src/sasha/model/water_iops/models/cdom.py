from ....model.water_iops.models_registry.cdom_registry import FUNCTION_REGISTRY_CDOM
from pydantic import Field
from ....model.common.core import SpectralComponentLoader
from typing import Optional, Callable



class CdomAbsorptionModel(SpectralComponentLoader):
    model_option: str = "bricaud"
    DEFAULT_CDOM: float = Field(default=-3)
    DEFAULT_SCDOM_VALUE: float = Field(default=0.014)
    _model_func: Optional[Callable] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.DEFAULT_CDOM = -3 if "log" in self.model_option else 1e-2
        self._initialize_model()
        self.logger.info(f"Initialized Cdom Absorption Model with model option: {self.model_option}")

    def _initialize_model(self):
        model_data = FUNCTION_REGISTRY_CDOM.get(self.model_option)
        if model_data is None or 'a' not in model_data or 'model' not in model_data['a']:
            raise ValueError(f"Invalid model option: {self.model_option}")
        self._model_func = model_data['a']['model']
        # Pre-convert wvl to the correct backend format

    def get_property(self, **kwargs):
        cdom = kwargs.get("cdom", self.DEFAULT_CDOM)
        S_cdom = kwargs.get("S_cdom", self.DEFAULT_SCDOM_VALUE)
        if cdom is None:
            raise ValueError("cdom must be provided")
        return self._model_func(cdom, S_cdom, self.wvl, backend=self.backend)



class CdomBackscatteringModel(SpectralComponentLoader):
    model_option: str = "None"
    DEFAULT_CDOM: float = Field(default=-3)
    _model_func: Optional[Callable] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.DEFAULT_CDOM = -3 if "log" in self.model_option else 1e-2
        self._initialize_model()
        self.logger.info(f"Initialized Cdom Backscattering Model with model option: {self.model_option}")

    def _initialize_model(self):
        # Pre-compute the zeros array
        self._zeros = self._cp.zeros(self.wvl.shape)
        # Set up the model function
        self._model_func = lambda *args, **kwargs: self._zeros

    def get_property(self, **kwargs):
        cdom = kwargs.get("cdom", self.DEFAULT_CDOM)
        if cdom is None:
            raise ValueError("cdom must be provided")
        # No need to convert cdom as it's not used in the current implementation
        return self._model_func()



# Usage example:
# cdom_absorption = CdomAbsorptionModel(wvl=some_wavelengths, model_option="bricaud", backend=Backend.NUMPY)
# result = cdom_absorption.get_property(cdom=0.5)