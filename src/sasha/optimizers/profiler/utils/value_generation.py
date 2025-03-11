# shallow_sa/optimizers/profiler/utils/value_generation.py
"""
Value generation utilities for profiling.
"""

import numpy as np
class ValueGenerator:
    """Handles generation of values for profiling analysis."""
    def __init__(self):
        pass
    
    def get_width(self, mle_value: float, std_ofim: float, width: float) -> float:
        """Get width for interest value generation."""
        if width == 0:
            if std_ofim > 0:
                return 3.14 * std_ofim
            else : 
                width   = 0.2 * mle_value
        return width
        
    def generate_interest_values_uniform(self, mle_value: float, num_points: int, 
                                       width: float, std_ofim: float) -> np.ndarray:
        """Generate uniformly spaced values around MLE estimate."""
        w = self.get_width(mle_value, std_ofim, width)
        return np.linspace(mle_value - w, mle_value + w, num=num_points)
        
    def generate_interest_values_diffuse(self, mle_value: float, num_points: int,
                                       width: float, std_ofim: float) -> np.ndarray:
        """Generate values with higher density near MLE estimate.
        Returns a sorted array of values, with higher sampling density near the MLE estimate
        and lower density towards the edges of the range.
        """
        w = self.get_width(mle_value, std_ofim, width)
        # Generate base linear spacing
        interest_vals = np.linspace(
            mle_value - w, mle_value + w, num=num_points
        )
        # Calculate center index
        center_index = num_points // 2
        # Create array with higher density in the middle
        # First half: Take every other value from start to center
        first_half = interest_vals[:center_index:2]
        # Second half: Take every other value from center to end
        second_half = interest_vals[center_index::2]
        # Remaining values: Take the skipped values
        remaining = interest_vals[1::2]
        # Combine and sort all values
        return np.sort(np.concatenate([first_half, second_half, remaining]))
    
    def generate_interest_values(self, mle_value: float, num_points: int,
                               width: float, std_ofim: float, values: str) -> np.ndarray:
        """Generate interest values using specified method."""
        if values == "uniform":
            return self.generate_interest_values_uniform(mle_value, num_points, width, std_ofim)
        return self.generate_interest_values_diffuse(mle_value, num_points, width, std_ofim)