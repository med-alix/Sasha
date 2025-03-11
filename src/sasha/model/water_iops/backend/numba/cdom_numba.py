import numpy as np
from numba import jit

LOG_EPS = 1e-18  # ensure numerical stability


@jit(nopython=True)
def log_rescale(cdom, epsilon=0):
    return np.log(cdom + epsilon)


@jit(nopython=True)
def log_inverse_rescale(cdom_scaled, epsilon=0):
    return np.exp(cdom_scaled) - epsilon


@jit(nopython=True)
def bricaud_cdom_model(cdom_array, S_cdom, wvl):
    exp_term = np.exp(-S_cdom * (wvl - 443))
    return cdom_array.reshape(-1, 1) * exp_term


@jit(nopython=True)
def bricaud_cdom_model_deriv(cdom_array, S_cdom, wvl):
    ones = np.exp(0 * cdom_array).reshape(-1, 1)
    exp_term = ones * np.exp(-S_cdom * (wvl - 443))
    return exp_term


@jit(nopython=True)
def bricaud_cdom_model_hessian(cdom_array, S_cdom, wvl):
    return np.zeros(
        (cdom_array.size, wvl.size, 1, 1)
    )  # Replace with your expected shape


@jit(nopython=True)
def log_bricaud_cdom_model(cdom_array, S_cdom, wvl):
    cdom_original = np.exp(cdom_array).reshape(-1, 1)
    exp_term = np.exp(-S_cdom * (wvl - 443))
    return cdom_original * exp_term


@jit(nopython=True)
def log_bricaud_cdom_model_deriv(cdom_array, S_cdom, wvl):
    extra_term = np.exp(cdom_array).reshape(-1, 1)
    exp_term = np.exp(-S_cdom * (wvl - 443))
    return extra_term * exp_term


@jit(nopython=True)
def log_bricaud_cdom_model_hessian(cdom_array, S_cdom, wvl):
    extra_term = np.exp(cdom_array).reshape(-1, 1)
    exp_term = np.exp(-S_cdom * (wvl - 443))
    hessian = extra_term * exp_term
    hessian = hessian.reshape(
        hessian.shape[0], hessian.shape[1], 1, 1
    )  # Shape it into a 4D array for consistency
    return hessian
