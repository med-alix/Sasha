import jax
import jax.numpy as jnp

@jax.jit
def sigmoid(wvl, k, d):
    return 1 / (1 + jnp.exp(-k * (wvl - d)))


@jax.jit
def bump_function(wvl, k1, d1):
    f1 = sigmoid(wvl, k1, d1)
    return 1 - f1


@jax.jit
def bricaud_model(chl, Aphy, Ephy, wvl):
    if jnp.isscalar(chl) :
        return Aphy * (chl ** Ephy)
    else:
        return Aphy * (chl.reshape(-1, 1) ** Ephy)


@jax.jit
def log_bricaud_model(chl, Aphy, Ephy, wvl):
    chl = jnp.exp(chl)
    if jnp.isscalar(chl) :
        return Aphy * (chl**Ephy)
    else : 
        return Aphy * (chl.reshape(-1, 1)**Ephy)


@jax.jit
def two_peaks_model(chl, a_1, a_2, lambda_1, lambda_2, sigma_1, sigma_2, p, k1, d1, wvl):
    if not jnp.isscalar(chl) :
        chl   = chl.reshape(-1, 1)
    bump      = 1 - 1 / (1 + jnp.exp(-k1 * (wvl - d1)))
    exp_term1 = a_1 * jnp.exp(-((wvl - lambda_1) ** 2) / (2 * sigma_1**2))
    exp_term2 = bump * a_2 * jnp.exp(-((wvl - lambda_2) ** 2) / (2 * sigma_2**2))
    return (chl ** p) * (exp_term1 + exp_term2)


@jax.jit
def micro_nano_model(chl_micro_array, chl_nano_array, abs_micro, abs_nano, wvl):
    return (chl_micro_array.reshape(-1, 1) * abs_micro + chl_nano_array.reshape(-1, 1) * abs_nano)

# bricaud_model = jax.jit(bricaud_model, static_argnums=(3,))
# log_bricaud_model = jax.jit(log_bricaud_model, static_argnums=(3,))

@jax.jit
def None_model(chl,wvl):
    if jnp.isscalar(chl) :
        return jnp.zeros_like(wvl)
    else:
        return jnp.zeros_like(chl.size,wvl.size)
    