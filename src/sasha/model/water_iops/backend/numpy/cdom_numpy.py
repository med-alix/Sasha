import numpy as np
from numba import jit


# @jit(nopython=True)
def bricaud_cdom_model(cdom, S_cdom, wvl):
    if np.isscalar(cdom) :
        return  cdom*np.exp(-S_cdom * (wvl - 443))
    else:
        exp_term = np.exp(-S_cdom * (wvl - 443))
        return cdom.reshape(-1, 1) * exp_term


# @jit(nopython=True)
def log_bricaud_cdom_model(cdom, S_cdom, wvl):
    if np.isscalar(cdom) :
        cdom_original= np.exp(cdom)
        return  cdom_original*np.exp(-S_cdom * (wvl - 443))
    else:
        exp_term = np.exp(-S_cdom * (wvl - 443))
        cdom_original = np.exp(cdom)
        exp_term = np.exp(-S_cdom * (wvl - 443))
        return cdom_original.reshape(-1, 1) * exp_term

