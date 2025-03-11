import numpy as np
from numba import jit


@jit(nopython=True)
def compute_bwater(wvl, Abwater, Ebwater):
    return 0.5 * Abwater * (wvl / 500.0) ** Ebwater
