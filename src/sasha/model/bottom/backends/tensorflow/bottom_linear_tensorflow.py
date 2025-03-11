import tensorflow as tf

@tf.function
def augment_alpha_bottom_mix(alpha_vals):
    alpha_vals = tf.convert_to_tensor(alpha_vals, dtype=tf.float32)  # Convert to tensor if not already
    missing_alpha_vals = 1 - tf.reduce_sum(alpha_vals, axis=0)
    # Concatenate the missing alpha values along the last dimension
    augmented_alpha = tf.concat([alpha_vals, [missing_alpha_vals]], axis=0)
    return tf.squeeze(augmented_alpha)


@tf.function
def compute_linear_albedo(alpha_vals, albedo_components, k_s):
    # Normalize so the coordinates sum to 1 along the 0th axis
    alpha_vals = tf.convert_to_tensor(alpha_vals, dtype=tf.float32)  # Ensure it's a tensor
    simple_coords = alpha_vals / tf.reduce_sum(alpha_vals, axis=0, keepdims=True)
    # Perform dot product if dimensions are aligned
    if simple_coords.shape[0] == albedo_components.shape[0]:
        albedo_components = tf.cast(albedo_components,tf.float32)
        return tf.tensordot(simple_coords, albedo_components, axes=[[0], [0]])
    else:
        raise ValueError(
            f"Shapes {simple_coords.shape} and {albedo_components.shape} cannot be broadcasted."
        )

@tf.function
def compute_sigmoid_albedo(alpha_vals, albedo_components, k_s):
    alpha_vals = tf.convert_to_tensor(alpha_vals, dtype=tf.float32)  # Ensure it's a tensor
    sigmoid_value = 1 / (1 + tf.exp(-alpha_vals * k_s))
    # Check if dimensions are aligned and then perform dot product
    if sigmoid_value.shape[0] == albedo_components.shape[0]:
        albedo_components = tf.cast(albedo_components,tf.float32)
        return tf.tensordot(sigmoid_value, albedo_components, axes=[[0], [0]])
    else:
        raise ValueError(
            f"Shapes {sigmoid_value.shape} and {albedo_components.shape} cannot be broadcasted."
        )
