# radiometry_registry.py
import tensorflow as tf
import jax.numpy as jnp
import numpy as np
import jax 
from ...config.backend import Backend


def integrate_reflectance_numpy(reflectance, response_matrix):
    reflectance = reflectance.T
    response_matrix_transposed = np.array(response_matrix).T
    integrated_reflectance = np.dot(response_matrix_transposed, reflectance)
    return integrated_reflectance.T


@jax.jit
def integrate_reflectance_jax(reflectance, response_matrix):    
    reflectance = reflectance.T
    response_matrix_transposed = jnp.array(response_matrix).T
    integrated_reflectance = jnp.dot(response_matrix_transposed, reflectance)
    return integrated_reflectance.T



@tf.function(input_signature=[
    tf.TensorSpec(shape=(None,None), dtype=tf.float32),
    tf.TensorSpec(shape=(None,None), dtype=tf.float32),
])
def integrate_reflectance_tensorflow(reflectance, response_matrix):
    reflectance = tf.transpose(reflectance)
    response_matrix_transposed = tf.transpose(response_matrix)
    integrated_reflectance = tf.linalg.matmul(response_matrix_transposed, reflectance)
    return tf.transpose(integrated_reflectance)


def get_integrate_reflectance(backend):
    if backend == Backend.NUMPY:
        return integrate_reflectance_numpy
    elif backend == Backend.JAX:
        return integrate_reflectance_jax
    elif backend == Backend.TENSORFLOW:
        return integrate_reflectance_tensorflow