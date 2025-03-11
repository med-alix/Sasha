from ....config.backend import Backend
import numpy as np
import jax.numpy as jnp
import tensorflow as tf


def get_sum_op(backend,axis=None):
    """Get the appropriate sum operation for the given backend."""
    if backend == Backend.TENSORFLOW:
        return lambda x: tf.reduce_sum(x, axis=axis)    
    elif backend == Backend.JAX:
        return lambda x: jnp.sum(x, axis=axis)
    elif backend == Backend.NUMPY:
        return lambda x: np.sum(x, axis=axis)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
def get_max_op(backend, axis=None):
    """Get the appropriate max operation for the given backend."""
    if backend == Backend.TENSORFLOW:
        return lambda x: tf.reduce_max(x)
    elif backend == Backend.JAX:
        return lambda x: jnp.max(x, axis=axis)
    elif backend == Backend.NUMPY:
        return lambda x: np.max(x,  axis=axis)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

def get_min_op(backend, axis=None):
    """Get the appropriate min operation for the given backend."""
    if backend == Backend.TENSORFLOW:
        return lambda x: tf.reduce_min(x)
    elif backend == Backend.JAX:
        return lambda x: jnp.min(x, axis=axis)
    elif backend == Backend.NUMPY:
        return lambda x: np.min(x,  axis=axis)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

def get_sqrt_op(backend):
    """Get the appropriate sqrt operation for the given backend."""
    if backend == Backend.TENSORFLOW:
        return lambda x: tf.sqrt(x)
    elif backend == Backend.JAX:
        return lambda x: jnp.sqrt(x)
    elif backend == Backend.NUMPY:
        return lambda x: np.sqrt(x)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

def get_log_op(backend):
    """Get the appropriate log operation for the given backend."""
    if backend == Backend.TENSORFLOW:
        return lambda x: tf.math.log(x)
    elif backend == Backend.JAX:
        return lambda x: jnp.log(x)
    elif backend == Backend.NUMPY:
        return lambda x: np.log(x)
    else:
        raise ValueError(f"Unsupported backend: {backend}")