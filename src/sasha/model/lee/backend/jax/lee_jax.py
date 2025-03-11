import jax
import jax.numpy as jnp

@jax.jit
def compute_rrsw(z, kc, rrs_dp):
    if not jnp.isscalar(z):
        z = z.reshape(-1, 1)
    rrsc = rrs_dp * (1.0 - jnp.exp(-kc * z))
    return rrsc


@jax.jit
def compute_rrsb(z, kb, albedo):
    if not jnp.isscalar(z):
        z = z.reshape(-1, 1)
    rrsb = (albedo / jnp.pi) * jnp.exp(-kb * z)
    return rrsb

@jax.jit
def compute_rrsm(z, kc, rrs_dp, kb, albedo):
    rrsw = compute_rrsw(z, kc, rrs_dp)
    rrsb = compute_rrsb(z, kb, albedo)
    return rrsw + rrsb

@jax.jit
def compute_rrsp(z, kc, rrs_dp, kb, albedo, nconv, dconv):
    rrsm = compute_rrsm(z, kc, rrs_dp, kb, albedo)
    rrsp = nconv * rrsm / (1.0 - dconv * rrsm)
    return rrsp

# compute_rrsw = jax.jit(compute_rrsw, static_argnums=(2,))
# compute_rrsb = jax.jit(compute_rrsb, static_argnums=(2,))
# compute_rrsm = jax.jit(compute_rrsm, static_argnums=(4,))
# compute_rrsp = jax.jit(compute_rrsp, static_argnums=(6,))

