
from .core import Spectral
from pydantic import Field
from typing import Dict, Union
from pydantic import field_validator
from pathlib import Path
import yaml



class SpectralComponentLoader(Spectral):
  
    """Class for loading and managing spectral components"""
    pure_water: Dict[str, Path] = Field(default_factory=dict)
    phytoplankton: Dict[str, Path] = Field(default_factory=dict)
    bottom_components: Dict[str, Path] = Field(default_factory=dict)
    def __init__(self, **data):
        super().__init__(**data)
        self.load_empirical_components()

    def load_empirical_components(self):
        """Load empirical components from configuration file"""
        base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        config_path = base_dir / 'empirical_components.yml'

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        base_path = base_dir / config['base_path']

        # Convert paths 
        self.pure_water = {
            key: base_path / str(path) 
            for key, path in config['water_components']['pure_water'].items()
        }
        self.phytoplankton = {
            key: base_path / str(path) 
            for key, path in config['water_components']['phytoplankton'].items()
        }
        self.bottom_components = {
            key: base_path / str(path) 
            for key, path in config['bottom_components'].items()
        }
        # Validate all paths after setting
        self.validate_paths()


    def validate_paths(self):
        """Validate all component paths"""
        for component_dict in [self.pure_water, self.phytoplankton, self.bottom_components]:
            for name, path in component_dict.items():
                if not path.exists() or not path.is_file():
                    raise FileNotFoundError(f"The file {path} for component {name} does not exist or is not a file.")
                try:
                    with open(path, 'r') as file:
                        pass
                except Exception as e:
                    raise ValueError(f"Cannot read the file {path} for component {name}: {e}")


    @field_validator('pure_water', 'phytoplankton', 'bottom_components')
    @classmethod
    def validate_component_paths(cls, v: Dict[str, Union[str, Path]]) -> Dict[str, Path]:
        """Validate dictionary of paths"""
        if not isinstance(v, dict):
            raise ValueError(f"Expected dictionary, got {type(v)}")
        
        validated = {}
        for key, path in v.items():
            if isinstance(path, (str, Path)):
                path = Path(str(path))
                validated[key] = path
            else:
                raise ValueError(f"Invalid path type for {key}: {type(path)}")
        
        return validated


