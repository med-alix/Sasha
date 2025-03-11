import numpy as np
from numba import jit


@jit(nopython=True)
def assemble_hessians(hessians_list):
    # Get the shape of the first Hessian to determine n and m
    if len(hessians_list) == 1:
        return hessians_list[0]
    n, m, _, _ = hessians_list[0].shape
    # Calculate the total number of parameters (p) with a running sum
    total_params = 0
    for h in hessians_list:
        total_params += h.shape[-1]
    # Initialize the full Hessian with zeros
    full_hessian = np.zeros((n, m, total_params, total_params))
    # Fill in the blocks
    start_idx = 0
    for h in hessians_list:
        end_idx = start_idx + h.shape[-1]
        full_hessian[:, :, start_idx:end_idx, start_idx:end_idx] = h
        start_idx = end_idx
    return full_hessian


@jit(nopython=True)
def compute_a(aphy, acdom, anap, awater):
    return aphy + acdom + anap + awater


@jit(nopython=True)
def compute_a_deriv(*args):
    for a in args:
        print(a.shape)
    return np.stack(args, axis=2)


# @jit(nopython=True)
def compute_a_hess(*args):
    return assemble_hessians(args)


#  bp, bw,Bfp
@jit(nopython=True)
def compute_bb(b_chl, b_cdom, b_tsm, b_pure):
    return b_tsm + b_pure + b_chl + b_cdom


# @jit(nopython=True)
def compute_bb_deriv(b_chl, b_cdom, b_tsm):
    # print('b_chl',b_chl.shape)
    # print('b_cdom',b_cdom.shape)
    # print('b_tsm',b_tsm.shape)
    return np.stack((b_chl, b_cdom, b_tsm), axis=2)


def compute_bb_hess(b_chl, b_cdom, b_tsm):
    return assemble_hessians([b_chl, b_cdom, b_tsm])


@jit(nopython=True)
def compute_k(a, bb):
    return a + bb


@jit(nopython=True)
def compute_k_deriv(a_deriv, bb_deriv):
    return a_deriv + bb_deriv


@jit(nopython=True)
def compute_k_hess(a_hess, bb_hess):
    return a_hess + bb_hess


@jit(nopython=True)
def compute_u(bb, k):
    return bb / k


@jit(nopython=True)
def compute_u_deriv(bb, k, bb_deriv, k_deriv):
    n, m, p = k_deriv.shape
    bb = bb.reshape((n, m, 1))
    k = k.reshape((n, m, 1))
    denominator = k**2
    numerator = bb_deriv * k - bb * k_deriv
    return numerator / denominator


@jit(nopython=True)
def compute_u_hess(bb, k, bb_deriv, k_deriv, bb_hess, k_hess):
    n, m, p = k_deriv.shape
    bb = bb.reshape((n, m, 1, 1))
    k = k.reshape((n, m, 1, 1))
    bb_deriv = bb_deriv.reshape((n, m, p, 1))
    k_deriv = k_deriv.reshape((n, m, p, 1))
    denominator = k**4
    numerator = k**2 * (bb_hess * k - bb * k_hess) - 2 * k * (
        bb_deriv * k - bb * k_deriv
    )
    return numerator / denominator
