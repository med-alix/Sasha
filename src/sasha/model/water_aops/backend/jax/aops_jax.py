import jax
import jax.numpy as jnp

# @jax.jit
def compute_kb(u, k, theta, thetaw, gamma_b, alpha_b, beta_b):
    Dub = gamma_b * jnp.sqrt(alpha_b + beta_b * u)
    return (1.0 / jnp.cos(thetaw) + Dub/ jnp.cos(theta)) * k

# @jax.jit
def compute_rrsdp(u, alpha_dp, beta_dp):
    return (alpha_dp + beta_dp * u) * u


compute_kb = jax.jit(compute_kb, static_argnums=(6,))
compute_rrsdp = jax.jit(compute_rrsdp, static_argnums=(2,))
