from typing import Dict, Callable, Tuple, List, Any
from pydantic import Field, PrivateAttr
from ....model.common.core import SpectralComponentLoader
from ....model.water_iops.models.iops import WaterIOPs
from ....model.water_aops.models_registry.aops_registry import FUNCTION_REGISTRY_AOPs
from ....config.backend import Backend, get_cp, convert_array
import numpy as np


class WaterAOPs(SpectralComponentLoader):
    aphy_model_option: str
    acdom_model_option: str
    anap_model_option: str
    bb_chl_model_option: str
    bb_cdom_model_option: str
    bb_tsm_model_option: str
    modeling_args: Dict[str, Any] = Field(default_factory=dict)

    _waterIOPs: WaterIOPs = PrivateAttr()
    _model_data: Dict[str, Callable] = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_components()

    def _initialize_components(self):
        self._waterIOPs = WaterIOPs(
            wvl=self.wvl,
            aphy_model_option=self.aphy_model_option,
            acdom_model_option=self.acdom_model_option,
            anap_model_option=self.anap_model_option,
            bb_chl_model_option=self.bb_chl_model_option,
            bb_cdom_model_option=self.bb_cdom_model_option,
            bb_tsm_model_option=self.bb_tsm_model_option,
            backend=self.backend,
            **self.modeling_args
        )
        self._model_data = FUNCTION_REGISTRY_AOPs
        if not self._model_data:
            raise ValueError("Model data registry is empty or undefined")

    def _compute_property(self, prop_type: str, **kwargs):
        method = getattr(self, f"_get_{prop_type}_args", None)
        if not method:
            raise ValueError(f"Unsupported property type: {prop_type}")
        args = method(**kwargs)
        model_func = self._model_data[prop_type]["model"]
        return model_func(*args, backend=self.backend)

    def _get_kc_args(self, **kwargs):
        theta_v, theta_s = self._get_geom_parameters(**kwargs)
        gamma_c, alpha_c, beta_c = self._get_kc_parameters(**kwargs)
        k, u = self.get_k_and_u(**kwargs)
        return [u, k, theta_v, theta_s, gamma_c, alpha_c, beta_c]

    def _get_kb_args(self, **kwargs):
        theta_v, theta_s = self._get_geom_parameters(**kwargs)
        gamma_b, alpha_b, beta_b = self._get_kb_parameters(**kwargs)
        k, u = self.get_k_and_u(**kwargs)
        return [u, k, theta_v, theta_s, gamma_b, alpha_b, beta_b]

    def _get_rrsdp_args(self, **kwargs):
        alpha_dp, beta_dp = self._get_rrsdp_parameters(**kwargs)
        _, u = self.get_k_and_u(**kwargs)
        return [u, alpha_dp, beta_dp]

    def _get_kc_parameters(self, **kwargs):
        return (
            kwargs.get("gamma_c", 1.03),
            kwargs.get("alpha_c", 1),
            kwargs.get("beta_c", 2.4)
        )

    def _get_kb_parameters(self, **kwargs):
        return (
            kwargs.get("gamma_b", 1.04),
            kwargs.get("alpha_b", 1),
            kwargs.get("beta_b", 5.4)
        )

    def _get_rrsdp_parameters(self, **kwargs):
        return (
            kwargs.get("alpha_dp", 0.084),
            kwargs.get("beta_dp", 0.170)
        )

    def _get_geom_parameters(self, **kwargs):
        return (
            kwargs.get("theta_v", 30 * np.pi / 180),
            kwargs.get("theta_s", 30 * np.pi / 180)
        )
    
    def get_k_and_u(self, **kwargs) -> Tuple[Any, Any]:
        k = self._waterIOPs.get_property(prop_type="k", **kwargs)
        u = self._waterIOPs.get_property(prop_type="u", **kwargs)
        return k, u

    def get_property(self, prop_type: str, **kwargs):
        return self._compute_property(prop_type, **kwargs)

    def get_multiple_properties(self, prop_types: List[str], **kwargs):
        return {prop_type: self.get_property(prop_type, **kwargs) for prop_type in prop_types}




# Usage example:
# water_aops = WaterAOPs(wvl=some_wavelengths, aphy_model_option="option1", ..., backend=Backend.NUMPY)
# result = water_aops.get_property(prop_type="kb", **some_kwargs)