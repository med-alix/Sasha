# backend.py
from   enum import Enum
import numpy as np



class Backend(Enum):
    JAX = "jax"
    NUMPY = "numpy"
    NUMBA = "numba"
    TENSORFLOW = "tensorflow"

class BackendManager:
    def __init__(self):
        self._backend = Backend.NUMPY
        self._cp = None
        self._ArrayType = None
        self._integrate_reflectance = None
        self._update_backend()

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value):
        if not isinstance(value, Backend):
            raise ValueError(f"Backend must be a Backend enum, got {type(value)}")
        self._backend = value
        self._update_backend()

    @property
    def cp(self):
        return self._cp

    @property
    def ArrayType(self):
        return self._ArrayType

    @property
    def integrate_reflectance(self):
        return self._integrate_reflectance

    def _update_backend(self):
        if self._backend == Backend.JAX:
            import jax.numpy as jnp
            self._cp = jnp
            self._ArrayType = jnp.ndarray
        elif self._backend in [Backend.NUMPY, Backend.NUMBA]:
            import numpy as np
            self._cp = np
            self._ArrayType = np.ndarray
        elif self._backend == Backend.TENSORFLOW:
            import tensorflow as tf
            self._cp = tf
            self._ArrayType = tf.Tensor
        else:
            raise ValueError(f"Invalid backend option: {self._backend}")

    def convert_array(self, array):
        if self._backend == Backend.TENSORFLOW:
            return self._cp.convert_to_tensor(array, dtype=self._cp.float32)
        else:
            return self._cp.array(array, dtype=self._cp.float32)




# Create a global instance of BackendManager
backend_manager = BackendManager()
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
import numba as numba_np

def get_ArrayType(backend):
    return backend_manager.ArrayType



def get_cp(backend):
    if backend == Backend.NUMPY:
        return np
    elif backend == Backend.JAX:
        return jnp
    elif backend == Backend.TENSORFLOW:
        return tf
    elif backend == Backend.NUMBA:
        return numba_np
    else:
        raise ValueError(f"Unsupported backend: {backend}")

def convert_array(array, backend):
    if backend == Backend.NUMPY:
        return np.array(array, dtype= np.float32)
    elif backend == Backend.JAX:
        return jnp.array(array, dtype= jnp.float32)
    elif backend == Backend.TENSORFLOW:
        return tf.convert_to_tensor(array, dtype  = tf.float32)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
