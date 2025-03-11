# Linear Albedo model with JAX and jit
import jax.numpy as jnp
from jax import jit


@jit
def augment_alpha_bottom_mix(alpha_vals):
    missing_alpha_vals = 1 - jnp.sum(jnp.array(alpha_vals), axis=0)
    # Concatenate the missing alpha values
    augmented_alpha = jnp.vstack(alpha_vals + [missing_alpha_vals])
    return augmented_alpha.squeeze()


@jit
def compute_linear_albedo(alpha_vals, albedo_components, k_s):
    # Normalize so the coordinates sum to 1 along the 0th axis
    simple_coords = alpha_vals / jnp.sum(alpha_vals, axis=0, keepdims=True)
    # Check if dimensions are aligned and then perform dot product
    if simple_coords.shape[0] == albedo_components.shape[0]:
        return jnp.dot(simple_coords.T, albedo_components)
    else:
        raise ValueError(
            f"Shapes {simple_coords.shape} and {albedo_components.shape} cannot be broadcasted."
        )


# Sigmoid-based Albedo model with JAX and ajit
@jit
def compute_sigmoid_albedo(alpha_vals, albedo_components, k_s):
    # Compute sigmoid values
    sigmoid_value = 1 / (1 + jnp.exp(-alpha_vals * k_s))
    # Check if dimensions are aligned and then perform dot product
    if sigmoid_value.shape[0] == albedo_components.shape[0]:
        return jnp.dot(sigmoid_value.T, albedo_components)
    else:
        raise ValueError(
            f"Shapes {sigmoid_value.shape} and {albedo_components.shape} cannot be broadcasted."
        )

