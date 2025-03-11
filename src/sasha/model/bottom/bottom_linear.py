from ...model.bottom.bottom_base import BottomBase
from ...model.bottom.models_registry.bottom_linear_registry import FUNCTION_REGISTRY_BOTTOM
from pydantic import  PrivateAttr
from ...config.backend import  get_cp, convert_array
from typing import Any



class BottomLinear(BottomBase):
    wvl: Any
    modeling_option: str
    DEFAULT_KS: float = 1e-1
    _model_function: callable = PrivateAttr()
    _end_members: Any = PrivateAttr()
    _n_members: int = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._cp = get_cp(self.backend)
        self._initialize_components()

    def _initialize_components(self):
        self._model_function = FUNCTION_REGISTRY_BOTTOM[self.modeling_option]["model"]
        self.setup_properties()

    def setup_properties(self):
        albedo_components = []
        albedo_files = [
            self.bottom_components['sand_albedo_file'],
            self.bottom_components['vege_albedo_file']
        ]

        for albedo_file in albedo_files:
            wvl_alb, alb = self._read_model_data(albedo_file)
            interp_alb = self._interp_data(wvl_alb, alb)
            albedo_components.append(interp_alb)

        self._end_members = self._cp.stack(albedo_components, axis=0)
        self._n_members = len(albedo_files)

    def get_property(self, **kwargs):
        augmented_alpha = self._augment_alpha_mix(**kwargs)
        k_s = convert_array(kwargs.get("k_s", self.DEFAULT_KS), self.backend)
        return self._model_function(
            augmented_alpha,
            self._end_members,
            k_s=k_s,
            backend=self.backend
        )

    def _augment_alpha_mix(self, **kwargs):
        alpha_vals = [
            convert_array(kwargs.get(f"alpha_m_{i+1}", None), self.backend)
            for i in range(self._n_members - 1)
        ]
        if not alpha_vals:
            raise ValueError(
                "alpha_m_i coefficient should match endmembers count minus 1"
            )
        return FUNCTION_REGISTRY_BOTTOM["augment_alpha_bottom_mix"](alpha_vals, backend=self.backend)

# Usage example:
# bottom_linear = BottomLinear(wvl=some_wavelengths, modeling_option="linear", backend=Backend.NUMPY)
# result = bottom_linear.get_property(k_s=0.2, alpha_m_1=0.5)