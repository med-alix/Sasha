from ...model.common.core import SpectralComponentLoader
from ...model.water_aops.models.aops import WaterAOPs
from ...model.bottom.bottom_linear import BottomLinear
from pydantic import PrivateAttr
from ...model.lee.models_registry.lee_registry import FUNCTION_REGISTRY_LEE
from typing import Dict, Any, List

class LeeModel(SpectralComponentLoader):
    aphy_model_option: str
    acdom_model_option: str
    anap_model_option: str
    bb_chl_model_option: str
    bb_cdom_model_option: str
    bb_tsm_model_option: str
    bottom_model_option: str

    _water: WaterAOPs = PrivateAttr()
    _bottom: BottomLinear = PrivateAttr()
    _model_data: Dict[str, Any] = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_components()

    def _initialize_components(self):
        """Initializes sub-components of the Lee model with provided wavelength and modeling arguments."""
        self._bottom = BottomLinear(
            wvl=self.wvl,
            modeling_option=self.bottom_model_option,
            backend=self.backend
        )
        self._water = WaterAOPs(
            wvl=self.wvl,
            aphy_model_option=self.aphy_model_option,
            acdom_model_option=self.acdom_model_option,
            anap_model_option=self.anap_model_option,
            bb_chl_model_option=self.bb_chl_model_option,
            bb_cdom_model_option=self.bb_cdom_model_option,
            bb_tsm_model_option=self.bb_tsm_model_option,
            backend=self.backend
        )
        self._model_data = FUNCTION_REGISTRY_LEE

    def _compute_property(self, prop_type: str, **kwargs):
        method_name = f"_get_{prop_type}_args"
        method = getattr(self, method_name, None)
        if not method:
            raise ValueError(f"Unsupported property type: {prop_type}")
        z = kwargs.get('z', None)
        args = [z] + method(**kwargs)
        model_func = self._model_data[prop_type]["model"]
        return model_func(*args, backend=self.backend)

    def _get_rrsb_args(self, **kwargs):
        kb = self._water.get_property("kb", **kwargs)
        albedo = self._bottom.get_property(**kwargs)
        return [kb, albedo]

    def _get_rrsw_args(self, **kwargs):
        kc = self._water.get_property("kc", **kwargs)
        rrsdp = self._water.get_property("rrsdp", **kwargs)
        return [kc, rrsdp]

    def _get_rrsm_args(self, **kwargs):
        kc = self._water.get_property("kc", **kwargs)
        rrsdp = self._water.get_property("rrsdp", **kwargs)
        kb = self._water.get_property("kb", **kwargs)
        albedo = self._bottom.get_property(**kwargs)
        return [kc, rrsdp, kb, albedo]

    def _get_rrsp_args(self, **kwargs):
        args = self._get_rrsm_args(**kwargs)
        surface_params = self._get_surface_parameters(**kwargs)
        return args + surface_params

    def _get_surface_parameters(self, **kwargs):
        numerator = kwargs.get("nconv", 0.5)
        denominator = kwargs.get("dconv", 1.5)
        return [numerator, denominator]

    def get_property(self, prop_type: str, **kwargs):
        return self._compute_property(prop_type, **kwargs)

    def get_multiple_properties(self, prop_types: List[str], **kwargs):
        return {prop_type: self.get_property(prop_type, **kwargs) for prop_type in prop_types}