import tensorflow as tf

@tf.function
def compute_bwater(wvl, Abwater, Ebwater):
    # Compute bwater
    return tf.cast(0.5 * Abwater * (wvl / 500.0) ** Ebwater, tf.float32)