"""
Serialization utilities for profiling results.
"""

import json
import numpy as np
from typing import Any, Dict
import jaxlib.xla_extension as jax_xla

class ResultSerializer:
    """Handles serialization of profiling results."""
    
    def save_to_json(self, data: Dict[str, Any], file_path: str) -> None:
        """
        Save profiling results to JSON file.
        
        Args:
            data: Results to save
            file_path: Path to save JSON file
        """
        def default(obj):
            """Custom JSON serializer for special types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, jax_xla.DeviceArray):
                return np.array(obj).tolist()
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(file_path, "w") as outfile:
            json.dump(data, outfile, default=default, indent=4)
