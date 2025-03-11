import numpy as np
from numba import jit

LOG_EPS = 1e-18  # ensure numerical stability


# @jit(nopython=True)
def sigmoid(self, wvl, k, d):
    return 1 / (1 + np.exp(-k * (wvl - d)))


# @jit(nopython=True)
def bump_function(wvl, k1, d1):
    f1 = sigmoid(wvl, k1, d1)
    return 1 - f1


# @jit(nopython=True)
def bricaud_model(chl, Aphy, Ephy, wvl):
    if np.isscalar(chl):
        return Aphy * (chl ** Ephy)
    else:
        return Aphy * (chl.reshape(-1, 1) ** Ephy)


# @jit(nopython=True)
def log_bricaud_model(chl, Aphy, Ephy, wvl, epsilon=LOG_EPS):
    # Inverse transform of chl_array
    if np.isscalar(chl):
        chl_original = np.exp(chl)
    else:
        chl_original = np.exp(chl.reshape(-1, 1))
    return Aphy * (chl_original**Ephy)


# @jit(nopython=True)
def two_peaks_model(chl, a_1, a_2, lambda_1, lambda_2, sigma_1, sigma_2, p, k1, d1, wvl):
    if not np.isscalar(chl):
        chl = chl.reshape(-1, 1)
    bump = 1 - 1 / (1 + np.exp(-k1 * (wvl - d1)))
    exp_term1 = a_1 * np.exp(-((wvl - lambda_1) ** 2) / (2 * sigma_1**2))
    exp_term2 = bump * a_2 * np.exp(-((wvl - lambda_2) ** 2) / (2 * sigma_2**2))
    return (chl** p) * (exp_term1 + exp_term2)


# @jit(nopython=True)
def micro_nano_model(chl_micro_array, chl_nano_array, abs_micro, abs_nano, wvl):
    return (
        chl_micro_array.reshape(-1, 1) * abs_micro
        + chl_nano_array.reshape(-1, 1) * abs_nano
    )