import numpy as np
from numba import jit



@jit(nopython=True)
def compute_bb(b_chl, b_cdom, b_tsm, b_pure):
    return b_tsm + b_pure + b_chl + b_cdom


@jit(nopython=True)
def compute_a(aphy, acdom, anap, awater):
    return aphy + acdom + anap + awater


@jit(nopython=True)
def compute_k(a, bb):
    return a + bb


@jit(nopython=True)
def compute_u(bb, k):
    return bb / k
