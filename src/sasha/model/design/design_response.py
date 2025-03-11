import json
from pathlib import Path
import logging
from copy import deepcopy
from scipy.stats import norm
from ...model.common.core import SpectralComponentLoader
from pydantic import Field
from typing import Dict, Any, Optional
from ...config.backend import Backend, convert_array
from ...model.design.model_registry.design_response_registry import *
import numpy as np


class DesignResponse(SpectralComponentLoader):

    wvl: Any
    model_option: str
    response_data: Optional[Dict[str, Any]] = Field(default=None)
    response_matrix: Optional[Any] = Field(default=None)
    interpolated_response: Dict[str, Any] = Field(default_factory=dict)
    n_bands: Optional[int] = Field(default=None)
    bands: Optional[Dict[str, float]] = Field(default=None)
    retained_bands: Optional[list] = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_response()
        self.logger.info(f"Response matrix initialization completed")
        
    def _initialize_response(self):

        if self.model_option not in ['dirac']:
            self.response_data = self.get_response_data()
            min_sampled_wvl, max_sampled_wvl = self._compute_adjusted_band_ranges()
            self.wvl = self.wvl[(self.wvl > min_sampled_wvl) & (self.wvl < max_sampled_wvl)]
            if 'fwhm' in self.response_data:
                self._create_gaussian_responses()
                self.bands = {f'B{i+1}': wvl for i, wvl in enumerate(self.response_data['wavelengths'])}
                self.n_bands = len(self.bands)
            else:
                self.n_bands = len(list(self.response_data.keys()))
                self.bands = self._compute_band_centers()
            self._interpolate_response()
            self._compute_response_matrix()
        else:
            self.n_bands = 1
            self.bands = {'B1': 0.0}
            self.interpolated_response['B1'] = convert_array([1.0], self.backend)
            self.response_matrix = convert_array([1.0], self.backend)



    def get_response_data(self):
        file_mapping = {
            "sentinel_2A": "SRF_MS_SENTINEL.json",
            "sentinel_2B": "SRF_MS_SENTINEL.json",
            "observer_2deg_cie93": "SRF_HUMAN.json",
            "hyspex_muosli": "SRF_HS.json"
        }
        json_file = file_mapping.get(self.model_option)
        if not json_file:
            raise ValueError(f"Unknown model option: {self.model_option}")
        self.logger.info(f"Loading response data for model: {self.model_option}")
        return self._read_json_data(self._locate_srf_data(json_file))

    def _read_json_data(self, file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
        return data[self.model_option]

    def _locate_srf_data(self, json_file):
        script_dir = Path(__file__).resolve().parent
        data_path  = script_dir / "SRF_DATA" / json_file
        if not data_path.is_file():
            logger.error(f"File not found: {data_path}")
            raise FileNotFoundError(f"No such file or directory: '{data_path}'")
        return data_path

    def _interpolate_response(self):
        """
        Interpolates the sensor's response function over the class's wavelength array.
        """
        response_dict = deepcopy(self.response_data["response"])  
        wavelengths   = deepcopy(self.response_data["wavelengths"])
        bands         = deepcopy(self.bands)
        retained_bands = []
        self.interpolated_response = {}
        bands_centers = self._compute_band_centers()
        for i, (band, response) in enumerate(response_dict.items()):
            if bands_centers[band] < max(self.wvl) and bands_centers[band] > min(self.wvl):
                self.interpolated_response[band] = self._interp_data( wavelengths, response, extra_flag=False)
                retained_bands.append(bands_centers[band])
        self.response_data["wavelenghts"] = self.wvl
        self.retained_bands = retained_bands

    def _compute_band_centers(self):
        band_centers = {}
        _sum = get_sum_op(self.backend)
        wavelengths = convert_array(self.response_data['wavelengths'], self.backend)
        for band, response in self.response_data['response'].items():
            response = convert_array(response, self.backend)
            center = _sum(wavelengths * response) / _sum(response)
            band_centers[band] = center
        return band_centers

    def _compute_band_widths(self):
        band_widths = {}
        _max = get_max_op(self.backend)
        _min = get_min_op(self.backend)
        wavelengths = convert_array(self.response_data['wavelengths'], self.backend)
        for band, response in self.response_data['response'].items():
            response = convert_array(response, self.backend)
            active_wavelengths = wavelengths[response > 0]
            width = (_max(active_wavelengths) - _min(active_wavelengths) 
                    if len(active_wavelengths) > 0 else 0)
            band_widths[band] = width
        return band_widths

    def _create_gaussian_responses(self):
        """Create Gaussian response functions for each band"""
        import tensorflow as tf
        print("Starting Gaussian response creation...")
        # Convert inputs and print shapes
        fwhm_data = convert_array(self.response_data['fwhm'], self.backend)
        band_wavelengths = convert_array(self.response_data['wavelengths'], self.backend)
        wvl = convert_array(self.wvl, self.backend)
        self.response_data['response'] = {}
        peak_response = 1.0
        _max  = get_max_op(self.backend)
        _sqrt = get_sqrt_op(self.backend)
        _log  = get_log_op(self.backend)

        if self.backend == Backend.TENSORFLOW:
            # Convert all inputs to TensorFlow tensors explicitly
            fwhm_data = tf.cast(fwhm_data, tf.float32)
            band_wavelengths = tf.cast(band_wavelengths, tf.float32)
            wvl = tf.cast(wvl, tf.float32)
            try:
                # Reshape tensors for broadcasting
                wvl_expanded = tf.expand_dims(wvl, 0)  # [1, n_wavelengths]
                centers_expanded = tf.expand_dims(band_wavelengths, 1)  # [n_bands, 1]
                # Compute sigma
                sigma = fwhm_data / (2.0 * tf.sqrt(2.0 * tf.math.log(2.0)))
                sigma_expanded = tf.expand_dims(sigma, 1)  # [n_bands, 1]
                # Compute differences and Gaussians
                diff = wvl_expanded - centers_expanded
                exp_term = -0.5 * tf.square(diff / sigma_expanded)
                responses = tf.exp(exp_term)
                # Normalize responses
                max_responses = tf.reduce_max(responses, axis=1, keepdims=True)
                scaled_responses = (responses / max_responses) * peak_response
                # Store results
                for i in range(band_wavelengths.shape[0]):
                    band_response = scaled_responses[i]
                    self.response_data['response'][f'B{i+1}'] = band_response
                print("Successfully stored all interpolated band responses")
            except Exception as e:
                print(f"Error in TensorFlow computation: {str(e)}")
                raise
        else:
            # Original implementation for other backends
            for i, (center_wvl, fwhm) in enumerate(zip(band_wavelengths, fwhm_data)):
                sigma = fwhm / (2 * _sqrt(2.0 * _log(2.0)))
                response = np.array(norm.pdf(np.array(wvl), 
                                        center_wvl, 
                                        sigma)).astype(np.float32)
                response = convert_array(response, self.backend)
                max_response = _max(response)
                scaled_response = (response / max_response) * peak_response
                self.response_data['response'][f'B{i+1}'] = scaled_response
        
        self.response_data['wavelengths'] = wvl
        print("Completed Gaussian response creation")

    def _compute_response_matrix(self):
        """Compute normalized response matrix with proper backend handling"""
        self.logger.info("Computing response matrix")
        # Convert list of responses to array/tensor using appropriate backend
        responses            = [val for val in self.interpolated_response.values()]
        if self.backend      == Backend.TENSORFLOW:
            response_matrix  = tf.transpose(tf.convert_to_tensor(responses, dtype=tf.float32))
            max_values       = tf.reduce_sum(response_matrix, axis=0)
        elif self.backend    == Backend.JAX:
            response_matrix  = jnp.array(responses).T
            max_values       = jnp.sum(response_matrix, axis=0)
        else:  # NUMPY
            response_matrix  = np.array(responses).T
            max_values       = np.sum(response_matrix, axis=0)
        # Normalize each band by its sum
        self.response_matrix = response_matrix / max_values

    def _compute_adjusted_band_ranges(self):
        if 'fwhm' in self.response_data:
            fwhm         = convert_array(self.response_data["fwhm"], self.backend)
            wavelengths  = convert_array(self.response_data["wavelengths"], self.backend)
            adjusted_min = wavelengths[0] - 2 * fwhm[0]
            adjusted_max = wavelengths[-1] + 2 * fwhm[-1]
        else:
            band_centers = self._compute_band_centers()
            band_widths  = self._compute_band_widths()
            min_center   = min(band_centers.values())
            max_center   = max(band_centers.values())
            min_band     = min(band_centers, key=lambda k: band_centers[k])
            max_band     = max(band_centers, key=lambda k: band_centers[k])
            adjusted_min = min_center - band_widths[min_band] / 2
            adjusted_max = max_center + band_widths[max_band] / 2
        return adjusted_min, adjusted_max

# Usage example:
# design_response = DesignResponse(wvl=some_wavelengths, model_option="sentinel_2A", backend=Backend.NUMPY)