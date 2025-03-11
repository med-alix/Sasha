import jax
import jax.numpy as jnp

@jax.jit
def bricaud_cdom_model(cdom, S_cdom, wvl):
    if jnp.isscalar(cdom) :
        return  cdom*jnp.exp(-S_cdom * (wvl - 443))
    elif jnp.isscalar(S_cdom):
        exp_term = jnp.exp(-S_cdom * (wvl - 443))
        return cdom.reshape(-1, 1) * exp_term
    else:
        return cdom.reshape(-1, 1) * jnp.exp(-S_cdom.reshape(-1,1) * (wvl - 443))


@jax.jit
def log_bricaud_cdom_model(cdom, S_cdom, wvl):
    if jnp.isscalar(cdom) :
        return  jnp.exp(cdom)*jnp.exp(-S_cdom * (wvl - 443))
    elif jnp.isscalar(S_cdom):
        exp_term      = jnp.exp(-S_cdom * (wvl - 443))
        return jnp.exp(cdom).reshape(-1, 1) * exp_term
    else:
        return jnp.exp(cdom).reshape(-1, 1) * jnp.exp(-S_cdom.reshape(-1,1) * (wvl - 443))




@jax.jit
def None_model(cdom,wvl):
    if jnp.isscalar(cdom) :
        return jnp.zeros_like(wvl)
    else:
        return jnp.zeros_like(cdom.size,wvl.size)