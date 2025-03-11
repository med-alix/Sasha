import numpy as np
from numba import jit


# Bricaud anap
# @jit(nopython=True)
def bricaud_anap_model(tsm, Anap, Snap, wvl):
    if np.isscalar(tsm):
        return tsm * Anap * np.exp(-Snap * (wvl - 443))
    else:
        return tsm.reshape(-1, 1) * Anap * np.exp(-Snap * (wvl - 443))


# Bricaud bp
# @jit(nopython=True)
def bricaud_bp_model(tsm, Abp, Ebp, Bfp, wvl):
    if np.isscalar(tsm):   
        return Bfp * tsm * Abp * (wvl / 555.0) ** Ebp
    else:
        return Bfp * tsm.reshape(-1, 1) * Abp * (wvl / 555.0) ** Ebp


# Log bricaud anap
# @jit(nopython=True)
def log_bricaud_anap_model(tsm, Anap, Snap, wvl):
    if np.isscalar(tsm):
        tsm_original = np.exp(tsm)
        exp_term = np.exp(-Snap * (wvl - 443))
        return tsm_original * Anap * exp_term
    else:
        tsm_original = np.exp(tsm)  # Inverse of log(x + epsilon)
        exp_term = np.exp(-Snap * (wvl - 443))
        return tsm_original.reshape(-1, 1) * Anap * exp_term



# Log bricaud bp
# @jit(nopython=True)
def log_bricaud_bp_model(tsm, Abp, Ebp, Bfp, wvl):
    if np.isscalar(tsm):
        tsm_original = np.exp(tsm)  # Inverse of log(x + epsilon)
        return Bfp * tsm_original * Abp * (wvl / 555.0) ** Ebp
    else:
        tsm_original = np.exp(tsm)  # Inverse of log(x + epsilon)
        return Bfp * tsm_original.reshape(-1, 1) * Abp * (wvl / 555.0) ** Ebp


