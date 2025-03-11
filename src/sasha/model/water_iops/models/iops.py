from typing import Dict, Callable, Optional, List
from pydantic import Field, PrivateAttr
from ....model.common.core import SpectralComponentLoader
from ....model.water_iops.models_registry.iops_registry import FUNCTION_REGISTRY_IOPs
from ....model.water_iops.models.chl import ChlAbsorptionModel, ChlBackscatteringModel
from ....model.water_iops.models.cdom import CdomAbsorptionModel, CdomBackscatteringModel
from ....model.water_iops.models.pure import WaterPure
from ....model.water_iops.models.tsm import TsmAbsorptionModel, TsmBackscatteringModel



class WaterIOPs(SpectralComponentLoader):
    # Model options as regular fields
    aphy_model_option: str = Field(...)
    acdom_model_option: str = Field(...)
    anap_model_option: str = Field(...)
    bb_chl_model_option: str = Field(...)
    bb_cdom_model_option: str = Field(...)
    bb_tsm_model_option: str = Field(...)
    
    # Use PrivateAttr for model data to avoid validation
    _model_data: Dict[str, Dict[str, Callable]] = PrivateAttr(default_factory=dict)
    
    # Component models as private attributes
    _a_chl: Optional[ChlAbsorptionModel] = PrivateAttr(default=None)
    _a_cdom: Optional[CdomAbsorptionModel] = PrivateAttr(default=None)
    _a_tsm: Optional[TsmAbsorptionModel] = PrivateAttr(default=None)
    _bb_chl: Optional[ChlBackscatteringModel] = PrivateAttr(default=None)
    _bb_cdom: Optional[CdomBackscatteringModel] = PrivateAttr(default=None)
    _bb_tsm: Optional[TsmBackscatteringModel] = PrivateAttr(default=None)
    _water_pure: Optional[WaterPure] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_components()
        self._model_factory()

    def _initialize_components(self):
        """Initialize all component models"""
        self._a_chl = ChlAbsorptionModel(
            wvl=self.wvl, 
            model_option=self.aphy_model_option, 
            backend=self.backend
        )
        self._a_cdom = CdomAbsorptionModel(
            wvl=self.wvl, 
            model_option=self.acdom_model_option, 
            backend=self.backend
        )
        self._a_tsm = TsmAbsorptionModel(
            wvl=self.wvl, 
            model_option=self.anap_model_option, 
            backend=self.backend
        )
        self._bb_chl = ChlBackscatteringModel(
            wvl=self.wvl, 
            model_option=self.bb_chl_model_option, 
            backend=self.backend
        )
        self._bb_cdom = CdomBackscatteringModel(
            wvl=self.wvl, 
            model_option=self.bb_cdom_model_option, 
            backend=self.backend
        )
        self._bb_tsm = TsmBackscatteringModel(
            wvl=self.wvl, 
            model_option=self.bb_tsm_model_option, 
            backend=self.backend
        )
        self._water_pure = WaterPure(
            wvl=self.wvl, 
            backend=self.backend
        )

    def _model_factory(self):
        """Initialize model data registry"""
        if not FUNCTION_REGISTRY_IOPs:
            raise ValueError("Model data registry is empty or undefined")
        self._model_data = FUNCTION_REGISTRY_IOPs
    
    def _compute_property(self, prop_type: str, **kwargs):
        """Compute a specific optical property"""
        method_name = f"_get_{prop_type}_args"
        method = getattr(self, method_name, None)
        if method is None:
            raise ValueError(f"Unsupported property type: {prop_type}")
        args = method(**kwargs)
        model_func = self._model_data[prop_type]["model"]
        return model_func(*args, backend=self.backend)

    def _get_a_args(self, **kwargs):
        """Get arguments for absorption computation"""
        a_chl = self._a_chl.get_property(**kwargs)
        a_cdom = self._a_cdom.get_property(**kwargs)
        a_tsm = self._a_tsm.get_property(**kwargs)
        a_water = self._water_pure.awater
        return [a_chl, a_cdom, a_tsm, a_water]

    def _get_bb_args(self, **kwargs):
        """Get arguments for backscattering computation"""
        bb_chl = self._bb_chl.get_property(**kwargs)
        bb_cdom = self._bb_cdom.get_property(**kwargs)
        bb_tsm = self._bb_tsm.get_property(**kwargs)
        bb_water = self._water_pure.get_bwater()
        return [bb_chl, bb_cdom, bb_tsm, bb_water]

    def _get_k_args(self, **kwargs):
        """Get arguments for attenuation computation"""
        a  = self.get_property(prop_type="a", **kwargs)
        bb = self.get_property(prop_type="bb", **kwargs)
        return [a, bb]

    def _get_u_args(self, **kwargs):
        """Get arguments for average cosine computation"""
        bb = self.get_property(prop_type="bb", **kwargs)
        k  = self.get_property(prop_type="k", **kwargs)
        return [bb, k]

    def get_property(self, prop_type: str, **kwargs):
        """Public method to compute any optical property"""
        return self._compute_property(prop_type, **kwargs)

    def get_multiple_properties(self, prop_types: List[str], **kwargs):
        """Compute multiple optical properties at once"""
        return {prop_type: self.get_property(prop_type, **kwargs) 
                for prop_type in prop_types}



# Usage example:
# water_iops = WaterIOPs(wvl=some_wavelengths, aphy_model_option="option1", ..., backend=Backend.NUMPY)
# result = water_iops.get_property(prop_type="a", **some_kwargs)