import numpy as np

# from numba import jit


# @jit(nopython=True)
def augment_alpha_bottom_mix(alpha_vals):
    missing_alpha_vals = 1 - np.sum(np.array(alpha_vals), axis=0)
    # Concatenate the missing alpha values
    augmented_alpha = np.vstack(alpha_vals + [missing_alpha_vals])
    return augmented_alpha.squeeze()


# @jit(nopython=True)
def compute_linear_albedo(alpha_vals, albedo_components, k_s):
    # Normalize so the coordinates sum to 1 along the 0th axis
    simple_coords = alpha_vals / np.sum(alpha_vals, axis=0, keepdims=True)
    # Check if dimensions are aligned and then perform dot product
    if simple_coords.shape[0] == albedo_components.shape[0]:
        return np.dot(simple_coords.T, albedo_components)
    else:
        raise ValueError(
            "Shapes {} and {} cannot be broadcasted.".format(
                simple_coords.shape, albedo_components.shape
            )
        )


# @jit(nopython=True)
def compute_sigmoid_albedo(alpha_vals, albedo_components, k_s):
    # Compute sigmoid values
    sigmoid_value = 1 / (1 + np.exp(-alpha_vals * k_s))
    # Check if dimensions are aligned and then perform dot product
    if sigmoid_value.shape[0] == albedo_components.shape[0]:
        return np.dot(sigmoid_value.T, albedo_components)
    else:
        raise ValueError(
            "Shapes {} and {} cannot be broadcasted.".format(
                sigmoid_value.shape, albedo_components.shape
            )
        )


# @jit(nopython=True)
def compute_linear_albedo_deriv(alpha_vals, albedo_components, k_s=None):
    # Derivative for the linear model is the difference between the max and min along the spectral dimension
    return np.sum(albedo_components, axis=0) - np.min(albedo_components, axis=0)


# @jit(nopython=True)
def compute_sigmoid_albedo_deriv(alpha_vals, albedo_components, k_s):
    # Compute sigmoid values
    sigmoid_value = 1 / (1 + np.exp(-alpha_vals * k_s))
    # Compute the derivative of the sigmoid function
    sigmoid_deriv = k_s * sigmoid_value * (1 - sigmoid_value)
    # Check if dimensions are aligned and then sum along the first axis
    if sigmoid_deriv.shape[0] == albedo_components.shape[0]:
        return np.dot(sigmoid_deriv.T, albedo_components)
    else:
        raise ValueError(
            "Shapes {} and {} cannot be broadcasted.".format(
                sigmoid_deriv.shape, albedo_components.shape
            )
        )
