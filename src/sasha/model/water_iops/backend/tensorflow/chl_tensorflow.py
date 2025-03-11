import tensorflow as tf

@tf.function
def bricaud_model(chl_array, Aphy, Ephy, wvl):
    return Aphy * (chl_array ** Ephy)

@tf.function
def log_bricaud_model(chl_array, Aphy, Ephy, wvl):
    chl_original = tf.exp(chl_array)
    chl_original = tf.cast(chl_original, tf.float32)
    Ephy = tf.cast(Ephy, tf.float32)
    Aphy = tf.cast(Aphy, tf.float32)
    # Reshape chl_original to [15, 1] for broadcasting
    chl_original = tf.reshape(chl_original, [-1, 1])
    return Aphy * (chl_original ** Ephy)



@tf.function
def two_peaks_model(chl_array, a_1, a_2, lambda_1, lambda_2, sigma_1, sigma_2, p, k1, d1, wvl):
    bump = 1 - 1 / (1 + tf.exp(-k1 * (wvl - d1)))
    exp_term1 = a_1 * tf.exp(-((wvl - lambda_1) ** 2) / (2 * sigma_1 ** 2))
    exp_term2 = bump * a_2 * tf.exp(-((wvl - lambda_2) ** 2) / (2 * sigma_2 ** 2))
    return (chl_array ** p) * (exp_term1 + exp_term2)



@tf.function
def micro_nano_model(chl_micro_array, chl_nano_array, abs_micro, abs_nano, wvl):
    return chl_micro_array * abs_micro + chl_nano_array * abs_nano