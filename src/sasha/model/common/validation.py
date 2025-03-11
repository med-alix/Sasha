
from pydantic import BaseModel
from typing import  Tuple
import numpy as np  



class SpectralValidation(BaseModel):
    """Class for spectral data validation and range checking"""
    
    @staticmethod
    def check_wavelength_range(data_wvl: np.ndarray, target_wvl: np.ndarray) -> Tuple[bool, bool, float, float]:
        """
        Check if target wavelengths are within the data wavelength range
        
        Returns:
            Tuple containing:
            - bool: True if extrapolation needed below range
            - bool: True if extrapolation needed above range
            - float: Percentage of values requiring extrapolation below
            - float: Percentage of values requiring extrapolation above
        """
        min_data, max_data = np.min(data_wvl), np.max(data_wvl)
        min_target, max_target = np.min(target_wvl), np.max(target_wvl)
        
        extrap_below = min_target < min_data
        extrap_above = max_target > max_data
        
        # Calculate percentage of values requiring extrapolation
        below_count = np.sum(target_wvl < min_data)
        above_count = np.sum(target_wvl > max_data)
        total_points = len(target_wvl)
        
        pct_below = (below_count / total_points) * 100 if below_count > 0 else 0
        pct_above = (above_count / total_points) * 100 if above_count > 0 else 0
        
        return extrap_below, extrap_above, pct_below, pct_above

    @staticmethod
    def validate_spectral_data(data: np.ndarray) -> Tuple[bool, float]:
        """
        Validate spectral data for NaN values and negative values
        
        Returns:
            Tuple containing:
            - bool: True if data contains NaN values
            - float: Percentage of NaN values
        """
        nan_mask = np.isnan(data)
        has_nan = np.any(nan_mask)
        pct_nan = (np.sum(nan_mask) / len(data)) * 100 if has_nan else 0
        
        return has_nan, pct_nan
