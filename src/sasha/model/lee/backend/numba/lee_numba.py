import numpy as np
from numba import jit


def compute_rrsw(z, kc, rrs_dp):
    z_reshaped = z.reshape(-1, 1)
    rrsc = rrs_dp * (1.0 - np.exp(-kc * z_reshaped))
    return rrsc


def compute_rrsb(z, kb, albedo):
    z_reshaped = z.reshape(-1, 1)
    rrsb = (albedo / np.pi) * np.exp(-kb * z_reshaped)
    return rrsb


def compute_rrsm(z, kc, rrs_dp, kb, albedo):
    rrsw = compute_rrsw(z, kc, rrs_dp)
    rrsb = compute_rrsb(z, kb, albedo)
    return rrsw + rrsb


def compute_rrsp(z, kc, rrs_dp, kb, albedo, nconv, dconv):
    rrsm = compute_rrsm(z, kc, rrs_dp, kb, albedo)
    rrsp = nconv * rrsm / (1.0 - dconv * rrsm)
    return rrsp
