import jax
import jax.numpy as jnp

LOG_EPS = 1e-18  # ensure numerical stability

@jax.jit
def log_rescale(cdom, epsilon=0):
    return jnp.log(cdom + epsilon)

@jax.jit
def log_inverse_rescale(cdom_scaled, epsilon=0):
    return jnp.exp(cdom_scaled) - epsilon

@jax.jit
def bricaud_anap_model(tsm, Anap, Snap, wvl):
    if jnp.isscalar(tsm) :
        return  tsm*Anap*jnp.exp(-Snap * (wvl - 443))
    else:
        exp_term = jnp.exp(-Snap * (wvl - 443))
        return tsm.reshape(-1, 1) * Anap * exp_term

@jax.jit
def bricaud_bp_model(tsm, Abp, Ebp, Bfp, wvl):
    if jnp.isscalar(tsm) :
        return Bfp * tsm * Abp * (wvl / 555.0) ** Ebp
    else:
        return Bfp * tsm.reshape(-1, 1) * Abp * (wvl / 555.0) ** Ebp

@jax.jit
def None_model(tsm,wvl):
    if jnp.isscalar(tsm) :
        return jnp.zeros_like(wvl)
    else:
        return jnp.zeros_like(tsm.size,wvl.size)
    



@jax.jit
def log_bricaud_anap_model(tsm, Anap, Snap, wvl):
    tsm = log_inverse_rescale(tsm)
    if jnp.isscalar(tsm) :
        return  tsm*Anap*jnp.exp(-Snap * (wvl - 443))
    else:
        exp_term = jnp.exp(-Snap * (wvl - 443))
        return tsm.reshape(-1, 1) * Anap * exp_term

@jax.jit
def log_bricaud_bp_model(tsm, Abp, Ebp, Bfp, wvl):
    tsm = log_inverse_rescale(tsm)
    if jnp.isscalar(tsm) :
          return Bfp * tsm * Abp * (wvl / 555.0) ** Ebp
    else: 
          return Bfp * tsm.reshape(-1, 1) * Abp * (wvl / 555.0) ** Ebp



