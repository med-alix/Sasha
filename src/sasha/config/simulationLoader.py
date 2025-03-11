import os
from   pathlib import Path
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_simulation_config_file(config_path=None):
    """
    Find the simulation configuration file with the suffix 'simulated_conditions.yml' in the following order:
    1. Specified path
    2. Environment variable
    3. Default locations
    """
    suffix = "simulated_conditions.yml"

    if config_path:
        path = Path(config_path)
        if path.is_file() and path.name.endswith(suffix):
            return path
        else:
            logger.warning(f"Specified simulation config file not found or incorrect: {path}")

    # Check environment variable
    env_path = os.environ.get('SIMULATION_CONFIG')
    if env_path:
        path = Path(env_path)
        if path.is_file() and path.name.endswith(suffix):
            return path
        else:
            logger.warning(f"Simulation config file from environment variable not found or incorrect: {path}")

    # Check default locations
    possible_locations = [
        Path.cwd() / suffix,
        Path.cwd().parent / suffix,
        Path(__file__).resolve().parent / suffix,
        Path(__file__).resolve().parent.parent / suffix,
    ]

    for location in possible_locations:
        if location.is_file():
            return location

    raise FileNotFoundError("Simulation configuration file not found")

def load_simulation_configuration(config_path=None):
    try:
        config_file = find_simulation_config_file(config_path)
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Simulation configuration loaded from: {config_file}")
        return config
    except FileNotFoundError as e:
        logger.error(f"Error loading simulation configuration: {e}")
        raise

class SimulationSet:
    def __init__(self):
        self.predefined_conditions = load_simulation_configuration()
    def reload_config(self, config_path=None):
        self.predefined_conditions = load_simulation_configuration(config_path)






# Create a global instance for simulation configuration
simulation_conditions = SimulationSet().predefined_conditions
