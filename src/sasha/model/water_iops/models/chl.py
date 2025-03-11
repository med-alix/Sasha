from pydantic import Field, PrivateAttr
from typing import Dict, Callable, Any
from ....model.common.core import SpectralComponentLoader
from ....model.water_iops.models_registry.chl_registry import FUNCTION_REGISTRY_APHY
from ....config.backend import  get_cp, convert_array

class ChlAbsorptionModel(SpectralComponentLoader):
    
    model_option: str = "bricaud"
    DEFAULT_CHL: float = Field(default=0.1)
    DEFAULT_TWO_PEAKS_PARAMS: Dict[str, float] = Field(default_factory=lambda: {
        "a_1": 0.03498720996860133, "a_2": 0.342155440335681,
        "lambda_1": 442.2871694159062, "lambda_2": 1038.7033934619485,
        "sigma_1": 55.44455553145052, "sigma_2": 147.97439804961323,
        "p": 0.6560867815061463, "k1": 0.2346079671431631,
        "d1": 688.1267296385715
    })

    _model_func: Callable = PrivateAttr()
    _property_data: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        self._cp = get_cp(self.backend)
        self._initialize_model()
        self._setup_properties()
        self.logger.info(f"Initialized Chl Absorption Model with model option: {self.model_option}")


    def _initialize_model(self):
        model_data = FUNCTION_REGISTRY_APHY.get(self.model_option)
        if model_data is None or 'model' not in model_data:
            raise ValueError(f"Invalid model option: {self.model_option}")
        self._model_func = model_data['model']

    def _setup_properties(self):
        property_files = {
            'Aphy': 'aphy_file',
            'Ephy': 'ephy_file',
            'abs_micro': 'abs_micro_file',
            'abs_nano': 'abs_nano_file'
        }
        for prop, file_key in property_files.items():
            self._property_data[prop] = self.initialize_property(self.phytoplankton[file_key])



    def get_property(self, **kwargs):
        chl = convert_array(kwargs.pop("chl", self.DEFAULT_CHL), self.backend)
        if self.model_option == "two_peaks":
            two_peaks_kwargs = {key: kwargs.get(key, val) for key, val in self.DEFAULT_TWO_PEAKS_PARAMS.items()}
            args = [chl] + list(two_peaks_kwargs.values()) + [self.wvl]
        elif self.model_option in ["bricaud", "log_bricaud"]:
            args = [chl, self._property_data['Aphy'], self._property_data['Ephy'], self.wvl]



        elif self.model_option == "micro_nano":
            chl_micro = convert_array(kwargs.pop("chl_micro", self.DEFAULT_CHL), self.backend)
            chl_nano = convert_array(kwargs.pop("chl_nano", self.DEFAULT_CHL), self.backend)
            args = [chl_micro, chl_nano, self._property_data['abs_micro'], self._property_data['abs_nano'], self.wvl]
        else:
            raise ValueError(f"Unsupported model option: {self.model_option}")
        return self._model_func(*args, backend=self.backend)



class ChlBackscatteringModel(SpectralComponentLoader):
    model_option: str = "None"
    DEFAULT_CHL: float = Field(default=-1)
    _zero_array: Any = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._cp = get_cp(self.backend)
        self.DEFAULT_CHL = -1 if "log" in self.model_option else 1e-2
        self._zero_array = self._cp.zeros((1, self.wvl.shape[0]))
        self.logger.info(f"Initialized Chl Backscattering Model with model option: {self.model_option}")

    def get_property(self, **kwargs):
        chl = convert_array(kwargs.get("chl", self.DEFAULT_CHL), self.backend)
        if chl is None:
            raise ValueError("chl must be provided")
        return self._zero_array

# Usage example:
# chl_absorption = ChlAbsorptionModel(wvl=some_wavelengths, model_option="bricaud", backend=Backend.NUMPY)
# result = chl_absorption.get_property(chl=0.5)