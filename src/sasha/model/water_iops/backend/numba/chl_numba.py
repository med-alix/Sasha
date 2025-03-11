import numpy as np
from numba import jit

LOG_EPS = 1e-18  # ensure numerical stability


@jit(nopython=True)
def sigmoid(self, wvl, k, d):
    return 1 / (1 + np.exp(-k * (wvl - d)))


@jit(nopython=True)
def bump_function(wvl, k1, d1):
    f1 = sigmoid(wvl, k1, d1)
    return 1 - f1


@jit(nopython=True)
def bricaud_model(chl_array, Aphy, Ephy, wvl):
    return Aphy * (chl_array.reshape(-1, 1) ** Ephy)


@jit(nopython=True)
def bricaud_model_deriv(chl_array, Aphy, Ephy, wvl):
    return Ephy * Aphy * (chl_array.reshape(-1, 1) ** (Ephy - 1))


@jit(nopython=True)
def bricaud_model_hessian(chl_array, Aphy, Ephy, wvl):
    # Compute the second derivative with respect to chl
    second_deriv = Ephy * (Ephy - 1) * Aphy * (chl_array.reshape(-1, 1) ** (Ephy - 2))
    # Since we only have one parameter (chl), the Hessian is just the second derivative
    # But we'll format it into a 4D array for consistency
    hessian_bricaud = second_deriv.reshape(
        second_deriv.shape[0], second_deriv.shape[1], 1, 1
    )
    return hessian_bricaud


# @jit(nopython=True)
def log_bricaud_model(chl_array, Aphy, Ephy, wvl, epsilon=LOG_EPS):
    # Inverse transform of chl_array
    chl_original = np.exp(chl_array.reshape(-1, 1))
    return Aphy * (chl_original**Ephy)


@jit(nopython=True)
def log_bricaud_model_deriv(chl_array, Aphy, Ephy, wvl):
    # Calculate the extra term for the derivative
    chl_term = np.exp(chl_array.reshape(-1, 1))
    # Compute the original model's derivative term
    original_term = Ephy * Aphy * (chl_term ** (Ephy))
    # Multiply by the extra term due to the log transformation
    return original_term


@jit(nopython=True)
def log_bricaud_model_hessian(chl_array, Aphy, Ephy, wvl):
    # Calculate the extra term for the derivative
    chl_term = np.exp(chl_array.reshape(-1, 1))
    # Compute the second derivative term
    second_deriv_term = Ephy * (Ephy - 1) * Aphy * (chl_term ** (Ephy))

    # Multiply by the extra term due to the exponential transformation
    # Format it into a 4D array for consistency
    hessian_log_bricaud = second_deriv_term.reshape(
        second_deriv_term.shape[0], second_deriv_term.shape[1], 1, 1
    )
    return hessian_log_bricaud


@jit(nopython=True)
def two_peaks_model(
    chl_array, a_1, a_2, lambda_1, lambda_2, sigma_1, sigma_2, p, k1, d1, wvl
):
    bump = 1 - 1 / (1 + np.exp(-k1 * (wvl - d1)))
    exp_term1 = a_1 * np.exp(-((wvl - lambda_1) ** 2) / (2 * sigma_1**2))
    exp_term2 = bump * a_2 * np.exp(-((wvl - lambda_2) ** 2) / (2 * sigma_2**2))
    return (chl_array.reshape(-1, 1) ** p) * (exp_term1 + exp_term2)


@jit(nopython=True)
def two_peaks_model_deriv(
    chl_array, a_1, a_2, lambda_1, lambda_2, sigma_1, sigma_2, p, k1, d1, wvl
):
    bump = 1 - 1 / (1 + np.exp(-k1 * (wvl - d1)))
    exp_term1 = a_1 * np.exp(-((wvl - lambda_1) ** 2) / (2 * sigma_1**2))
    exp_term2 = bump * a_2 * np.exp(-((wvl - lambda_2) ** 2) / (2 * sigma_2**2))
    ones_array = np.ones((chl_array.size, wvl.size))
    return (
        p * chl_array.reshape(-1, 1) ** (p - 1) * ones_array * (exp_term1 + exp_term2)
    )


@jit(nopython=True)
def two_peaks_model_hessian(
    chl_array, a_1, a_2, lambda_1, lambda_2, sigma_1, sigma_2, p, k1, d1, wvl
):
    # Calculate the bump term
    bump = 1 - 1 / (1 + np.exp(-k1 * (wvl - d1)))
    # Calculate the exponential terms
    exp_term1 = a_1 * np.exp(-((wvl - lambda_1) ** 2) / (2 * sigma_1**2))
    exp_term2 = bump * a_2 * np.exp(-((wvl - lambda_2) ** 2) / (2 * sigma_2**2))
    # Compute the second derivative term
    second_deriv_term = (
        p * (p - 1) * (chl_array.reshape(-1, 1) ** (p - 2)) * (exp_term1 + exp_term2)
    )
    # Format it into a 4D array for consistency
    hessian_two_peaks = second_deriv_term.reshape(
        second_deriv_term.shape[0], second_deriv_term.shape[1], 1, 1
    )
    return hessian_two_peaks


@jit(nopython=True)
def micro_nano_model(chl_micro_array, chl_nano_array, abs_micro, abs_nano, wvl):
    return (
        chl_micro_array.reshape(-1, 1) * abs_micro
        + chl_nano_array.reshape(-1, 1) * abs_nano
    )


@jit(nopython=True)
def micro_nano_model_deriv(chl_micro_array, chl_nano_array, abs_micro, abs_nano, wvl):
    ones_array_micro = np.ones((chl_micro_array.size, abs_micro.size))
    ones_array_nano = np.ones((chl_nano_array.size, abs_nano.size))
    return np.vstack((ones_array_micro * abs_micro, ones_array_nano * abs_nano))


@jit(nopython=True)
def micro_nano_model_hessian(chl_micro_array, chl_nano_array, abs_micro, abs_nano, wvl):
    # Initialize zero array for the Hessian
    n_replica = chl_micro_array.size  # Number of data points
    n_wvl = abs_micro.size  # Number of wavelengths
    n_params = 2  # Number of parameters (micro and nano)
    # Create a zero-filled Hessian array
    hessian_micro_nano = np.zeros((n_replica, n_wvl, n_params, n_params))
    return hessian_micro_nano
