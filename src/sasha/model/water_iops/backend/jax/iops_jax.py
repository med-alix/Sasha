import jax.numpy as jnp
from jax import jit



@jit
def compute_a(aphy, acdom, anap, awater):
    return aphy + acdom + anap + awater

@jit
def compute_bb(b_chl, b_cdom, b_tsm, b_pure):
    return b_tsm + b_pure + b_chl + b_cdom

@jit
def compute_k(a, bb):
    return a + bb

@jit
def compute_u(bb, k):
    return bb / k


