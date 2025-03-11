from pydantic import  Field, PrivateAttr, ConfigDict
import pandas as pd
from scipy.interpolate import Akima1DInterpolator, InterpolatedUnivariateSpline
from typing import Any, Tuple
import numpy as np
from ...config.backend import  backend_manager, get_cp, convert_array
from .base import BaseLogger
from .validation import SpectralValidation


class Spectral(BaseLogger):
    
    """base class for spectral data handling with validation and logging"""
    wvl: Any = Field(..., description="Array of wavelengths for the spectral data.")
    backend: Any = Field(default_factory=lambda: backend_manager.backend)
    extrapolation_threshold: float = Field(
        default=5.0,
        description="Maximum allowed percentage of values requiring extrapolation"
    )
    
    _cp: Any = PrivateAttr()
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
        validate_assignment=True,
        extra='allow'
    )
    
    def __init__(self, **data):
        if 'logger_name' not in data:
            data['logger_name'] = self.__class__.__name__
        super().__init__(**data)
        self.wvl = convert_array(self.wvl, self.backend)
        self._cp = get_cp(self.backend)
    
    @property
    def n_wvl(self) -> int:
        """Number of wavelength points"""
        return self._cp.shape(self.wvl)[0]

    def _read_model_data(self, datafile: str) -> Tuple[Any, Any]:
        """
        Read model data from file with enhanced error handling and validation
        """
        try:
            df = pd.read_csv(datafile, sep=r'\s+', header=None)
            df = df.dropna()
            
            if df.empty:
                raise ValueError(f"No valid data found in file: {datafile}")
                
            wvl = convert_array(df.iloc[:, 0].values, self.backend)
            data = convert_array(df.iloc[:, 1].values, self.backend)
            
            # Convert to numpy for validation
            wvl_np = np.array(wvl)
            data_np = np.array(data)
            
            # Check for monotonicity in wavelength
            if not np.all(np.diff(wvl_np) > 0):
                raise ValueError(f"Wavelength data in {datafile} is not strictly increasing")
            
            # Validate spectral data
            has_nan, pct_nan = SpectralValidation.validate_spectral_data(data_np)
            if has_nan:
                self.logger.warning(
                    f"Data from {datafile} contains {pct_nan:.2f}% NaN values"
                )
            
            return wvl, data
            
        except Exception as e:
            self.logger.error(f"Error reading file {datafile}: {str(e)}")
            raise

    def _interp_data(self, wvl: Any, data: Any, extra_flag: bool = False) -> Any:
        """
        Interpolate data with enhanced validation and range checking
        """
        # Convert to numpy for validation and interpolation
        wvl_np = np.array(wvl).astype(np.float32)
        data_np = np.array(data).astype(np.float32)
        target_wvl_np = np.array(self.wvl).astype(np.float32)
        
        # Check wavelength ranges
        extrap_below, extrap_above, pct_below, pct_above = SpectralValidation.check_wavelength_range(
            wvl_np, target_wvl_np
        )
        
        total_extrap = pct_below + pct_above
        if total_extrap > 0:
            msg = f"Wavelength range requires extrapolation: {pct_below:.1f}% below, {pct_above:.1f}% above"
            if total_extrap > self.extrapolation_threshold:
                self.logger.warning(f"{msg} (exceeds threshold of {self.extrapolation_threshold}%)")
            else:
                self.logger.info(msg)
        
        # Perform interpolation
        interpolator = Akima1DInterpolator(wvl_np, data_np)
        data_new = interpolator(target_wvl_np)
        
        # Handle extrapolation if needed
        has_nan = np.any(np.isnan(data_new))
        if has_nan and (extra_flag or total_extrap > 0):
            self.logger.info("Applying extrapolation to handle out-of-range values")
            extrapolator = InterpolatedUnivariateSpline(wvl_np, data_np, ext=3)
            data_new = extrapolator(target_wvl_np)
            
            # Validate results after extrapolation
            has_nan_after, pct_nan_after = SpectralValidation.validate_spectral_data(data_new)
            if has_nan_after:
                self.logger.error(
                    f"Interpolated data still contains {pct_nan_after:.2f}% NaN values after extrapolation"
                )
        
        return convert_array(data_new.astype(np.float32), self.backend)

    def resample(self, init_sp: Any, wvl: Any, fwhm: Any) -> Any:
        """
        Resample spectral data with validation
        """
        # Validate inputs
        for name, data in [("init_sp", init_sp), ("wvl", wvl), ("fwhm", fwhm)]:
            has_nan, pct_nan = SpectralValidation.validate_spectral_data(np.array(data))
            if has_nan:
                self.logger.warning(f"Input {name} contains {pct_nan:.2f}% NaN values")
        
        resamp_data = self._cp.zeros(self._cp.shape(wvl))
        dwvl = self.wvl[1] - self.wvl[0]
        
        for i_wvl in range(self._cp.shape(wvl)[0]):
            sigma = fwhm[i_wvl] / (2.0 * self._cp.sqrt(2 * self._cp.log(2)))
            filter_coeff = (1.0 / (sigma * self._cp.sqrt(2 * self._cp.pi)) * 
                          self._cp.exp(-0.5 * ((self.wvl - wvl[i_wvl]) ** 2) / (sigma ** 2)))
            filter_coeff *= dwvl
            resamp_data = resamp_data.at[i_wvl].set(self._cp.sum(filter_coeff * init_sp))
        
        return resamp_data

    def initialize_property(self, file_path: str) -> Any:
        """
        Initialize a spectral property from file with validation
        """
        self.logger.info(f"Initializing property from {file_path}")
        wvl_data, data = self._read_model_data(file_path)
        return self._interp_data(wvl_data, data)