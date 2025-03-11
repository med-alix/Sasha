import numpy as np
from scipy.interpolate import interp1d


def interpolate_nans_with_method(arr, method="linear"):
    """
    Interpolate NaN values in a 1D NumPy array using specified method, with a safeguard
    for empty or fully NaN arrays.
    Parameters:
    - arr (np.ndarray): The 1D array containing NaN values to interpolate.
    - method (str): The interpolation method ('linear', 'cubic', etc.)
    Returns:
    - np.ndarray: The array with NaN values interpolated, or an array of zeros if
    interpolation cannot be performed.
    """
    # Indices of the array
    x = np.arange(len(arr))
    # Indices of non-nan values
    non_nan_idx = np.where(~np.isnan(arr))[0]
    # Check if there are any non-NaN values to interpolate
    if len(non_nan_idx) == 0:
        # Return an array of zeros if there are no non-NaN values
        return np.zeros_like(arr)
    # Values of non-nan elements
    non_nan_vals = arr[non_nan_idx]
    # Create interpolation function
    f = interp1d(
        non_nan_idx,
        non_nan_vals,
        kind=method,
        bounds_error=False,
        fill_value="extrapolate",
    )
    # Interpolate NaN elements
    arr_new = np.copy(arr)  # Make a copy to avoid changing the original array
    nan_idx = np.isnan(arr)
    arr_new[nan_idx] = f(np.where(nan_idx)[0])
    return arr_new


def replace_NaN(initial_data):
    return np.array(
        [np.nan if not value else value for value in np.array(initial_data)]
    )
