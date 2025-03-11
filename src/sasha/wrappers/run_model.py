import yaml
import numpy as np
from ..model.common.core import Spectral
from ..model.radiometry.radiometry import Radiometry
from ..model.design.design_base import Design
from ..model.lee.lee_model import LeeModel
from ..config.backend import backend_manager, get_cp, Backend
from ..utils.array_utils import ensure_array_param_dict
import matplotlib.pyplot as plt
from ..model.common.base import BaseLogger
import logging
from pydantic import Field, PrivateAttr, ConfigDict
from typing import Any, Dict, Optional, ClassVar

custom_colors = {
    'primary': '#e30513',
    'secondary': '#ffc103',
    'accent1': '#0099ff',
    'background': '#2a2a2a',
    'text': '#ffffff',
    'grid': '#ffffff',
    'line': '#ffffff'  # Line color in white for contrast
}
plt.rcParams.update({
    # 'font.family': 'Roboto',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.unicode_minus': False,
    'text.color': custom_colors['text'],
    'axes.labelcolor': custom_colors['text'],
    'xtick.color': custom_colors['text'],
    'ytick.color': custom_colors['text'],
    'axes.titlecolor': custom_colors['text'],
})
plt.style.use('dark_background')



class ReflectanceSimulator(BaseLogger):

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types

    wvl: Optional[Any] = Field(default=None, description="Array of wavelengths for the spectral data.")
    backend: Any = Field(default_factory=lambda: backend_manager.backend)
    _cp: Any = PrivateAttr()
    config: Optional[Dict] = Field(default=None, description="Configuration dictionary for the simulator.")
    # Define fields for components
    design: Optional[Design] = Field(default=None, description="Design component for the simulator.")
    lee_model: Optional[LeeModel] = Field(default=None, description="Lee model component for the simulator.")
    radiometry: Optional[Radiometry] = Field(default=None, description="Radiometry component for the simulator.")



    def __init__(self, config_path=None, **kwargs):
        """Initialize analyzer with optional config path and kwargs for overrides"""
        super().__init__(logger_name="ReflectanceSimulator", log_level=kwargs.get('log_level', logging.INFO))

        try:
            self.config = self._load_config(config_path, **kwargs)
            self.setup_simulator()
        except Exception as e:
            raise

    def _load_config(self, config_path=None, **kwargs):
        """Load configuration from YAML file or use defaults, with kwargs overriding values."""
        
        # Start with default configuration
        config = {
            'backend': 'JAX',
            'design': {
                'sensor': 'hyspex_muosli',
                'min_wvl': 400,
                'max_wvl': 800,
                'wvl_step': 1
            },
            'modeling_args': {
                'aphy_model_option': 'log_bricaud',
                'acdom_model_option': 'log_bricaud',
                'anap_model_option': 'log_bricaud',
                'bb_chl_model_option': 'None',
                'bb_cdom_model_option': 'None',
                'bb_tsm_model_option': 'log_bricaud',
                'bottom_model_option': 'linear'
            }
        }
        
        try:
            # If config file provided, update with its values
            if config_path:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    config.update(file_config)
            
            # Override values from kwargs
            if kwargs:
                for key, value in kwargs.items():
                    if key == 'sensor':
                        config['design']['sensor'] = value
                    elif key.endswith('_model_option'):
                        config['modeling_args'][key] = value
                    else:
                        config[key] = value
            
            return config
            
        except Exception as e:
            raise
    
    def setup_simulator(self):
        """Setup analyzer components based on configuration"""
        
        try:
            # Setup backend
            self.backend = getattr(Backend, self.config['backend'])
            self._cp = get_cp(backend=self.backend)
            
            # Setup wavelength range
            self.wvl = np.arange(
                self.config['design']['min_wvl'],
                self.config['design']['max_wvl'],
                self.config['design']['wvl_step']
            )
            
            # Initialize components
            self.design = Design(
                model_option=self.config['design']['sensor'],
                wvl=self.wvl,
                backend=self.backend
            )
            
            self.lee_model = LeeModel(
                wvl=self.design.wvl,
                backend=self.backend,
                **self.config['modeling_args']
            )
            
            self.radiometry = Radiometry(
                design=self.design,
                response_matrix=self.design.response_matrix,
                reflectance_model=self.lee_model,
                prop_type=self.config['prop_type'],
                backend=self.backend
            )
                        
        except Exception as e:
            raise
    
    def simulate(self, params):
        """Simulate reflectance for given parameters"""
        
        try:
            sim_params = ensure_array_param_dict(params, backend=self.backend)
            
            reflectance = np.array(self.radiometry.sampled_reflectance(**sim_params)).T

            return reflectance
            
        except Exception as e:
            raise

    def simulate_native(self, params):
        """Simulate reflectance for given parameters"""
        sim_params = ensure_array_param_dict(params, backend=self.backend)
        return self.radiometry.sampled_reflectance(**sim_params)

    def plot(self, reflectance, title="Spectral Response", save_path=None):
        """Plot and optionally save the spectral response"""
        
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.design.bands, reflectance)
            plt.title(title)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Remote sensing reflectance (sr⁻¹)')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path)
                
            plt.show()
            
        except Exception as e:
            raise