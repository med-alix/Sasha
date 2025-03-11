import numpy as np
from numba import jit


@jit(nopython=True)
def compute_kb(u, k, theta, thetaw, gamma_b, alpha_b, beta_b):
    Dub = gamma_b * np.sqrt(alpha_b + beta_b * u)
    return (1.0 / np.cos(thetaw) + Dub) * k


@jit(nopython=True)
def compute_kb_jacob(k, u, k_deriv, u_deriv, theta, thetaw, gamma_b, alpha_b, beta_b):
    # Partial derivatives of kb with respect to k and u
    n, m, p = u_deriv.shape
    k = k.reshape(n, m, 1)
    u = u.reshape(n, m, 1)
    partial_kb_k = 1.0 / np.cos(thetaw) + gamma_b * np.sqrt(alpha_b + beta_b * u)
    partial_kb_u = (gamma_b * beta_b * k) / (2 * np.sqrt(alpha_b + beta_b * u))
    # Applying the chain rule
    kb_jacob = partial_kb_k * k_deriv + partial_kb_u * u_deriv
    return kb_jacob


# @jit(nopython=True)
@jit(nopython=True)
def compute_kb_hess(
    k, u, k_deriv, u_deriv, k_hess, u_hess, theta, thetaw, gamma_b, alpha_b, beta_b
):
    n, m, p, _ = u_hess.shape
    Dub = gamma_b * np.sqrt(alpha_b + beta_b * u)
    # Initialize Hessian tensor
    kb_hess = np.zeros((n, m, p, p))
    # First and second-order derivatives
    partial_kb_k = 1 / np.cos(thetaw) + Dub / np.cos(theta)
    partial_kb_u = -(gamma_b * beta_b * k) / (
        2 * np.sqrt(alpha_b + beta_b * u) * np.cos(theta)
    )
    partial_kb_kk = np.zeros((n, m))
    partial_kb_uu = -(gamma_b * beta_b * k) / (4 * np.power(alpha_b + beta_b * u, 1.5))
    partial_kb_ku = (gamma_b * beta_b) / (2 * np.sqrt(alpha_b + beta_b * u))
    # Manually calculate the outer product using broadcasting
    for i in range(p):
        for j in range(p):
            kb_hess[:, :, i, j] = (
                partial_kb_kk * k_deriv[:, :, i] * k_deriv[:, :, j]
                + partial_kb_uu * u_deriv[:, :, i] * u_deriv[:, :, j]
                + partial_kb_ku
                * (
                    k_deriv[:, :, i] * u_deriv[:, :, j]
                    + u_deriv[:, :, i] * k_deriv[:, :, j]
                )
            )
    # Adding the Hessians of k and u
    partial_kb_k = partial_kb_k.reshape(n, m, 1, 1)
    partial_kb_u = partial_kb_u.reshape(n, m, 1, 1)
    kb_hess += partial_kb_k * k_hess + partial_kb_u * u_hess
    return kb_hess


@jit(nopython=True)
def compute_rrsdp(u, alpha_dp, beta_dp):
    return (alpha_dp + beta_dp * u) * u


@jit(nopython=True)
def compute_rrsdp_jacob(u, alpha_dp, beta_dp, u_deriv):
    n, m, p = u_deriv.shape
    u = u.reshape(n, m, 1)
    # First-order derivative with respect to u
    partial_rrsdp_u = alpha_dp + 2 * beta_dp * u
    # Applying the chain rule for the Jacobian
    return partial_rrsdp_u * u_deriv


@jit(nopython=True)
def compute_rrsdp_hess(u, alpha_dp, beta_dp, u_deriv, u_hess):
    n, m, p, _ = u_hess.shape
    # Initialize the Hessian tensor
    u_deriv = u_deriv.reshape((n, m, p, 1))
    u = u.reshape(n, m, 1, 1)

    partial_rrsdp_u = alpha_dp + 2 * beta_dp * u
    # Second-order derivative with respect to u
    partial_rrsdp_uu = 2 * beta_dp  # This is a scalar
    # Manually calculate the outer product using broadcasting
    rrsdp_hess = partial_rrsdp_uu * u_deriv + partial_rrsdp_u * u_hess
    # Adding the Hessian of u (make sure the shapes are compatible)
    return rrsdp_hess
