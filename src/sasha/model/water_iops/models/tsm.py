from pydantic import Field
from ....model.common.core import SpectralComponentLoader
from typing import Optional, Callable, Any
from ....model.water_iops.models_registry.tsm_registry import FUNCTION_REGISTRY_TSM_ANAP, FUNCTION_REGISTRY_TSM_BP

class TsmAbsorptionModel(SpectralComponentLoader):
    model_option: str = "bricaud"
    DEFAULT_TSM: float = Field(default=-3)
    _model_func: Optional[Callable] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.DEFAULT_TSM = -3 if "log" in self.model_option else 0.1
        self._initialize_model()
        self.logger.info(f"Initialized Tsm Absorption Model with model option: {self.model_option}")

    def _initialize_model(self):
        model_data = FUNCTION_REGISTRY_TSM_ANAP.get(self.model_option, None)
        if model_data is None:
            if self.model_option in ["None"] : 
                self._model_func = lambda *args, **kwargs: self._cp.zeros((1, self.wvl.shape[0]))
            else :
                raise ValueError(f"Invalid model option: {self.model_option}")
        else:
            self._model_func = model_data['model']


    def get_property(self, deriv: bool = False, **kwargs) -> Any:
        Anap = kwargs.get("Anap", 0.036)
        Snap = kwargs.get("Snap", 0.0123)
        tsm = kwargs.get("tsm", self.DEFAULT_TSM)
        return self._model_func(tsm, Anap, Snap, self.wvl, backend=self.backend)




class TsmBackscatteringModel(SpectralComponentLoader):
    model_option: str = "None"
    DEFAULT_TSM: float = Field(default=-3)
    _model_func: Optional[Callable] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.DEFAULT_TSM = -3 if "log" in self.model_option else 0.1
        self._initialize_model()
        self.logger.info(f"Initialized Tsm Backscattering Model with model option: {self.model_option}")

    def _initialize_model(self):
        model_data = FUNCTION_REGISTRY_TSM_BP.get(self.model_option, None)
        if model_data is None:
            if self.model_option in ["None"] : 
                self._model_func = lambda *args, **kwargs: self._cp.zeros((1, self.wvl.shape[0]))
            else :
                raise ValueError(f"Invalid model option: {self.model_option}")
        else: 
            self._model_func = model_data['model']

    def get_property(self, deriv: bool = False, **kwargs) -> Any:
        Abp = kwargs.get("Abp", 0.42)
        Ebp = kwargs.get("Ebp", -0.2)
        Bfp = kwargs.get("Bfp", 0.0183)
        tsm = kwargs.get("tsm", self.DEFAULT_TSM)
        return self._model_func(tsm, Abp, Ebp, Bfp, self.wvl, backend=self.backend)

# Usage example:
# tsm_absorption = TsmAbsorptionModel(wvl=some_wavelengths, model_option="bricaud", backend=Backend.NUMPY)
# result = tsm_absorption.get_property(tsm=0.5)