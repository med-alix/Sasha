import os
import yaml
from pathlib import Path
from ..model.common.base import BaseLogger

# Set up logging

def find_config_file(config_path=None):
    """
    Find the configuration file in the following order:
    1. Specified path
    2. Environment variable
    3. Default locations
    """
    if config_path:
        path = Path(config_path)
        if path.is_file():
            return path
        else:
            raise FileNotFoundError(f"Configuration file not found at specified path: {path}")
    # Check environment variable
    env_path = os.environ.get('SHALLOW_SA_CONFIG')
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return path
        else:
            raise FileNotFoundError(f"Configuration file not found at path specified in environment variable SHALLOW_SA_CONFIG: {path}")
        

    # Check default locations
    possible_locations = [
        Path.cwd() / 'config_model.yml',
        Path.cwd().parent / 'config_model.yml',
        Path(__file__).parent.parent.parent / 'config_model.yml',
    ]

    for location in possible_locations:
        if location.is_file():
            return location

    raise FileNotFoundError("Configuration file not found")

def load_configuration(config_path=None):
    try:
        config_file = find_config_file(config_path)
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError as e:
        raise FileNotFoundError("Configuration file not found") from e
    


class GlobalConfig(BaseLogger):
    """Global configuration class that loads the configuration file and provides access to the settings"""
    def __init__(self):
        super().__init__(logger_name="GlobalConfig")
        self.config = load_configuration()
        self.update_from_config()

    def update_from_config(self):
        self.backend = self.config.get("backend", "JAX")
        self.deriv_backend = self.config.get("deriv_backend", "numerical")
        self.logger.info(f"Configured backend for computations: {self.backend}")
        self.logger.info(f"Configured derivative backend: {self.deriv_backend}")

    def reload_config(self, config_path=None):
        self.config = load_configuration(config_path)
        self.update_from_config()

# Create a global instance
# global_config = GlobalConfig()

