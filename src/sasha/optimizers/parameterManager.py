import os
from pathlib import Path
import yaml
import logging
from typing import Dict, Any, Tuple
import numpy as np 
from sasha.config.simulationLoader import simulation_conditions

class ParameterManager:
    """Manages parameter configurations for optimization."""
    
    def __init__(self, config: Dict[str, Any], 
                 param_file_path: str = None,
                 logger: logging.Logger = None):
        """
        Initialize ParameterManager with configuration.
        
        Args:
            config: Configuration dictionary containing parameterization settings
            param_file_path: Optional path to parameters YAML file. If None, uses default path
            logger: Optional logger instance. If None, creates a new one
        """
        self.activation_flags = config["parameterization"]["activation_flags"]
        self.modelling_args   = config["modeling_args"]
        self.parametrization  = config["simulation"]["parameterization"]
        self.parameters = {}  # Holds all parameters with their full specifications
        
        # Setup logging
        self.logger = logger or self._setup_logger()
        
        # Setup parameter file path
        if param_file_path is None:
            # Get the directory where this file is located
            current_dir = Path(__file__).parent
            # Navigate to config directory relative to this location
            default_path = current_dir.parent / "config" / "param_space.yml"
            self.param_file_path = str(default_path.resolve())
        else:
            self.param_file_path = param_file_path
            
        self.load_parameters()
        self.apply_transformations()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup a logger for the ParameterManager."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_parameters(self) -> None:
        """Load parameters from YAML file."""
        try:
            self.logger.debug(f"Attempting to load parameters from: {self.param_file_path}")
            
            with open(self.param_file_path, "r") as file:
                param_data = yaml.safe_load(file)
                
                for name, specs in param_data.items():
                    self.parameters[name] = {
                        "bounds": (specs["lower"], specs["upper"]),
                        "initial_value": specs["initial"],
                        "activated": self.activation_flags.get(name, False),
                    }
                    
            self.logger.info(f"Successfully loaded {len(param_data)} parameters")
            
        except FileNotFoundError:
            self.logger.error(f"Parameter file not found: {self.param_file_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML from file {self.param_file_path}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading parameters: {str(e)}")
            raise

    def apply_transformations(self):
        modelling_key_mapping = {
            "aphy_model_option": [
                "chl"
            ],  # Log transformation if 'log' is in aphy_model_option
            "acdom_model_option": ["cdom"],
            "anap_model_option": ["tsm"],
            "bb_chl_model_option": ["tsm"],
            "bb_cdom_model_option": ["cdom"],
            "bb_tsm_model_option": ["tsm"],
            "bottom_model_option": [
                "alpha_m_"
            ],  # Sigmoid transformation for parameters starting with 'alpha_m_'
        }

        for modelling_key, affected_params in modelling_key_mapping.items():
            option = self.modelling_args.get(modelling_key, "")
            for param in self.parameters:
                if any(param.startswith(prefix) for prefix in affected_params):
                    if "log" in option:
                        (
                            self.parameters[param]["initial_value"],
                            self.parameters[param]["bounds"],
                        ) = self._apply_log_transform(
                            self.parameters[param]["initial_value"],
                            self.parameters[param]["bounds"],
                        )
                    if "sigmoid" in option and modelling_key == "bottom_model_option":
                        (
                            self.parameters[param]["initial_value"],
                            self.parameters[param]["bounds"],
                        ) = self._apply_sigmoid_transform(
                            self.parameters[param]["initial_value"],
                            self.parameters[param]["bounds"],
                        )

    def substitute_fixed_param(self, param_to_substitute, value=None):
        """
        Fixes a parameter at a specific value or releases it back for optimization.

        Args:
        - param_to_substitute (str): The name of the parameter to be substituted.
        - value (float or None): The value to fix the parameter at. If None, the parameter
                                is released back for optimization.
        """
        if param_to_substitute in self.parameters:
            if value is not None:
                # Fix the parameter by deactivating it and setting it to a specific value
                self.parameters[param_to_substitute]["activated"] = False
                self.parameters[param_to_substitute]["fixed_value"] = value
            else:
                # Release the parameter for optimization
                if "fixed_value" in self.parameters[param_to_substitute]:
                    del self.parameters[param_to_substitute]["fixed_value"]
                self.parameters[param_to_substitute]["activated"] = True
        else:
            print(f"Parameter {param_to_substitute} not found.")

    def get_optimization_parameters(self):
        # Retrieve parameters separated into free and fixed sets for optimization
        free_params = {
            name: details
            for name, details in self.parameters.items()
            if details["activated"]
        }
        fixed_params = {
            name: details
            for name, details in self.parameters.items()
            if not details["activated"]
        }
        return free_params, fixed_params

    def transform_parameters(self, params_dict):
        # Define the mapping between modelling keys and the parameters they affect
        modelling_key_mapping = {
            "aphy_model_option": ["chl"],
            "acdom_model_option": ["cdom"],
            "anap_model_option": ["tsm"],
            "bb_chl_model_option": ["tsm"],
            "bb_cdom_model_option": ["cdom"],
            "bb_tsm_model_option": ["tsm"],
            "bottom_model_option": [
                "alpha_m_"
            ], 
        }
        def inverse_sigmoid(y):
            return -np.log((1 / y) - 1)
        for modelling_key, affected_params in modelling_key_mapping.items():
            option = self.modelling_args.get(modelling_key, "")
            for param in params_dict.keys():
                if "log" in option and param in affected_params:
                    params_dict[param] = np.log(max(params_dict[param], 1e-16))
                if "sigmoid" in option and "alpha_" in param:
                    params_dict[param] = inverse_sigmoid(params_dict[param])
        return params_dict

    def _apply_log_transform(self, initial_value, bounds):
        epsilon = 1e-16
        return np.log(max(initial_value, epsilon)), (
            np.log(max(bounds[0], epsilon)),
            np.log(max(bounds[1], epsilon)),
        )

    def _apply_sigmoid_transform(self, initial_value, bounds):
        def inverse_sigmoid(y):
            return -10 * np.log((1 / max(y, epsilon)) - 1)

        epsilon = 1e-10
        return inverse_sigmoid(initial_value), (
            inverse_sigmoid(bounds[0]),
            inverse_sigmoid(bounds[1]),
        )

    def update_parameter_values(self, params_update):
        """
        Updates the values of parameters based on a provided dictionary.

        Args:
        - params_update (dict): A dictionary with parameter names as keys and new values as values.
        """
        if params_update:
            for param, new_value in params_update.items():
                if param in self.parameters:
                    self.parameters[param]["initial_value"] = new_value
                else:
                    print(f"Warning: {param} not found in parameters.")
        else:
            pass
    def get_inactive_parameters(self, param_user):
        """
        Retrieves inactive parameters that are present in param_user.
        :param param_user: Dictionary of true values for all parameters.
        :return: Dictionary of inactive parameters intersected with param_user.
        """
        inactive_params = {
            name: self.parameters[name]
            for name in self.parameters
            if not self.parameters[name]["activated"] and name in param_user
        }
        # Return only the values from param_user for the inactive parameters
        return {name: param_user[name] for name in inactive_params}

    def get_param_from_config(self, config, z=5):
        # Implementation adjusted to be an internal method
        selected_params = simulation_conditions["predefined"][self.parametrization]
        param_config = {
            "chl":  selected_params["chl"],
            "cdom": selected_params["cdom"],
            "tsm":  selected_params["tsm"],
            "alpha_m_1": selected_params["alpha_m_1"],
            "z": float(z),
        }
        self.transform_parameters(param_config)
        return param_config

    def reduce_model(self, eta, config):
        """
        Creates a wrapper for the eta function that automatically incorporates
        inactive parameters fixed to their values in param_user.
        :param eta: Original eta function to be wrapped.
        :param parameter_manager: Instance of ParameterManager managing parameter activations.
        :param param_user: Dictionary with true parameter values.
        :return: Wrapped eta function.
        """
        param_config       = self.get_param_from_config(config)

        def eta_wrapper(**active_params):
            # Retrieve inactive parameters that are also in param_user
            inactive_params = self.get_inactive_parameters(param_config)
            # Merge active parameters with the fixed inactive ones for the call
            all_params = {**inactive_params, **active_params}
            return eta(**all_params)
        return eta_wrapper
