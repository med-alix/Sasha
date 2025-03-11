import numpy as np
from numba import jit

LOG_EPS = 1e-18  # ensure numerical stability


@jit(nopython=True)
def log_rescale(cdom, epsilon=0):
    return np.log(cdom + epsilon)


@jit(nopython=True)
def log_inverse_rescale(cdom_scaled, epsilon=0):
    return np.exp(cdom_scaled) - epsilon


# Bricaud anap
@jit(nopython=True)
def bricaud_anap_model(tsm_array, Anap, Snap, wvl):
    exp_term = np.exp(-Snap * (wvl - 443))
    return tsm_array.reshape(-1, 1) * Anap * exp_term


@jit(nopython=True)
def bricaud_anap_model_deriv(tsm_array, Anap, Snap, wvl):
    ones_array = np.ones((tsm_array.size, wvl.size))
    return ones_array * Anap * np.exp(-Snap * (wvl - 443))


# Bricaud bp
@jit(nopython=True)
def bricaud_bp_model(tsm_array, Abp, Ebp, Bfp, wvl):
    return Bfp * tsm_array.reshape(-1, 1) * Abp * (wvl / 555.0) ** Ebp


@jit(nopython=True)
def bricaud_bp_model_deriv(tsm_array, Abp, Ebp, Bfp, wvl):
    ones_array = np.ones((tsm_array.size, wvl.size))
    return Bfp * ones_array * Abp * (wvl / 555.0) ** Ebp


# Log bricaud anap
@jit(nopython=True)
def log_bricaud_anap_model(tsm_array, Anap, Snap, wvl):
    tsm_original = log_inverse_rescale(tsm_array)  # Inverse of log(x + epsilon)
    exp_term = np.exp(-Snap * (wvl - 443))
    return tsm_original.reshape(-1, 1) * Anap * exp_term


@jit(nopython=True)
def log_bricaud_anap_model_deriv(tsm_array, Anap, Snap, wvl):
    tsm_original = np.exp(tsm_array.reshape(-1, 1))  # Inverse of log(x + epsilon)
    # extra_term = np.exp(tsm_original)
    exp_term = np.exp(-Snap * (wvl - 443))
    exp_term_reshaped = exp_term.reshape(1, -1)
    deriv = tsm_original * Anap * exp_term_reshaped
    return deriv


@jit(nopython=True)
def log_bricaud_anap_model_hessian(tsm_array, Anap, Snap, wvl):
    tsm_original = np.exp(tsm_array.reshape(-1, 1))  # Inverse of log(x + epsilon)
    # extra_term = np.exp(tsm_original)
    exp_term = np.exp(-Snap * (wvl - 443))
    exp_term_reshaped = exp_term.reshape(1, -1)
    hess = tsm_original * Anap * exp_term_reshaped
    n, m = hess.shape
    return hess.reshape(n, m, 1, 1)


# Log bricaud bp
@jit(nopython=True)
def log_bricaud_bp_model(tsm_array, Abp, Ebp, Bfp, wvl):
    tsm_original = np.exp(tsm_array)  # Inverse of log(x + epsilon)
    return Bfp * tsm_original.reshape(-1, 1) * Abp * (wvl / 555.0) ** Ebp


@jit(nopython=True)
def log_bricaud_bp_model_deriv(tsm_array, Abp, Ebp, Bfp, wvl):
    extra_term = np.exp(tsm_array.reshape(-1, 1))
    bp_term = Bfp * Abp * (wvl / 555.0) ** Ebp
    bp_term_reshaped = bp_term.reshape(1, -1)
    deriv = extra_term * bp_term_reshaped
    return deriv


@jit(nopython=True)
def log_bricaud_bp_model_hessian(tsm_array, Abp, Ebp, Bfp, wvl):
    # Calculate the original tsm value
    extra_term = np.exp(tsm_array.reshape(-1, 1))
    # Calculate the bp term
    bp_term = Bfp * Abp * (wvl / 555.0) ** Ebp
    # Calculate the second derivative term
    second_deriv_term = extra_term * bp_term
    # Format it into a 4D array for consistency
    hessian_log_bp = second_deriv_term.reshape(
        second_deriv_term.shape[0], second_deriv_term.shape[1], 1, 1
    )
    return hessian_log_bp
