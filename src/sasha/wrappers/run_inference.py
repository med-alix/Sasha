import numpy as np
from functools import wraps
import time
from ..model.radiometry.radiometry import Radiometry
from ..model.design.design_base import Design
from ..model.lee.lee_model import LeeModel
from ..lk_inference.homoscedastic_model import HMOMVG
from ..optimizers.parameterManager import ParameterManager
from ..optimizers.optimizer import Optimizer
from ..optimizers.profiler import Profiler
from ..optimizers.profiler.visualization.plotting import ProfilePlotter
from   ..config.backend import backend_manager, get_cp, Backend
from ..model.common.base import BaseLogger
import yaml
import logging
from pydantic import Field, PrivateAttr, ConfigDict
from typing import Any, Dict, Optional, ClassVar


class ShallowSAInference(BaseLogger):


    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types
    wvl: Optional[Any] = Field(default=None, description="Array of wavelengths for the spectral data.")
    backend: Any = Field(default_factory=lambda: backend_manager.backend)
    _cp: Any = PrivateAttr()
    config: Optional[Dict] = Field(default=None, description="Configuration dictionary for the inference.")
    # Define fields for components
    design: Optional[Design] = Field(default=None, description="Design component for the inference.")
    lee_model: Optional[LeeModel] = Field(default=None, description="Lee model component for the inference.")
    radiometry: Optional[Radiometry] = Field(default=None, description="Radiometry component for the inference.")
    param_manager: Optional[ParameterManager] = Field(default=None, description="Parameter manager for the inference.")
    eta: Optional[Any] = Field(default=None, description="Reduced model for the inference.")
    mvg_inference: Optional[HMOMVG] = Field(default=None, description="Inference model for the inference.")
    optimizer: Optional[Optimizer] = Field(default=None, description="Optimizer for the inference.")


    def __init__(self, config_path=None,  **kwargs):
        super().__init__(logger_name="ShallowSAInference", log_level=kwargs.get('log_level', logging.INFO))
        try:
            # Load configuration
            self.config = self._load_config(config_path, **kwargs)
            self.setup_components()
        except Exception as e:
            raise Exception(f"Error in __init__: {str(e)}")

    def setup_components(self):
        """Setup all necessary components based on configuration."""        
        try:
            # Setup design
            config_design = self.config['design']
            self.backend = getattr(Backend, self.config['backend'])
            self._cp = get_cp(backend=self.backend)

            self.wvl = np.arange(
                self.config['design']['min_wvl'],
                self.config['design']['max_wvl'],
                self.config['design']['wvl_step']
            )
            
            
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
            # Setup parameter management and optimization components
            self.param_manager = ParameterManager(self.config)
            self.eta           = self.param_manager.reduce_model(self.radiometry.sampled_reflectance, config=self.config)
            
            # Setup inference model
            self.mvg_inference = HMOMVG(
                self.eta,
                (10**self.config['simulation']['std_noise_log'])**2,
                backend=self.backend
            )
            # Setup optimizer
            self.optimizer = Optimizer(
                stat_inf_obj=self.mvg_inference,
                parameter_manager=self.param_manager,
                method=self.config['optimizer']['method']
            )

        except Exception as e:
            raise Exception(f"Error in setup_components: {str(e)}")

    def simulate_reflectance(self, params=None):
        """Simulate reflectance using provided or default parameters."""
        try:
            params = params 
            reflectance = self.radiometry.sampled_reflectance(**params)
            return reflectance
        except Exception as e:
            raise Exception(f"Error in simulate_reflectance: {str(e)}")

    def generate_sample(self, params=None, n_samples=1):
        """Generate samples using the MVG inference model."""
        try:
            samples = self.mvg_inference.generate_samples(n_samples, **params)
            return samples.reshape(n_samples, -1)
        except Exception as e:
            raise Exception(f"Error in generate_sample: {str(e)}")

    def run_optimization(self, sample=None, **kwargs):
        """Run optimization on a given or generated sample."""
        try:
            if sample is None:
                sample = self.generate_sample(1)[0]
            result = self.optimizer.optimize(sample, **kwargs)
            return result
        except Exception as e:
            raise  Exception(f"Error in run_optimization: {str(e)}")

    def profile_parameter(self, param_name, sample=None, values = "uniform",
                           num_points=15, width=0.0, init_param = None, verbose=1):
        """Profile a specific parameter around the optimal value."""
        try:
            if sample is None:
                sample = self.generate_sample(1)[0]
            
            profiler      = Profiler(self.optimizer)
            profiled_data = profiler.profile(
                param_name,
                sample,
                values=values,
                num_points=num_points,
                width=width,
                init_param = init_param,
                verbose=verbose
            )
            return profiled_data
        except Exception as e:
            raise Exception(f"Error in profile_parameter: {str(e)}")

    def plot_profile(self, profiled_data):
        """Plot the profile results."""
        ProfilePlotter.plot_profile(profiled_data)


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
            raise Exception(f"Error in _load_config: {str(e)}")
    

