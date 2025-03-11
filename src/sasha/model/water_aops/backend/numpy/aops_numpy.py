import numpy as np
from numba import jit

# theta  : Subsurface is the subsurface viewing angle fromnadir
# thetaw : Subsurface surface solar zenith angle
@jit(nopython=True)
def compute_kb(u, k, theta, thetaw, gamma_b, alpha_b, beta_b):
    Dub = gamma_b * np.sqrt(alpha_b + beta_b * u)
    return (1.0 / np.cos(thetaw) + Dub / np.cos(theta)) * k


@jit(nopython=True)
def compute_rrsdp(u, alpha_dp, beta_dp):
    return (alpha_dp + beta_dp * u) * u

