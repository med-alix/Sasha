import jax.numpy as jnp
from jax import jit


@jit
def compute_bwater(wvl, Abwater, Ebwater):
    return 0.5 * Abwater * (wvl / 500.0) ** Ebwater


