import tensorflow as tf



@tf.function
def bricaud_anap_model(tsm_array, Anap, Snap, wvl):
    tsm_array = tf.expand_dims(tsm_array, axis=-1)
    exp_term = tf.exp(-Snap * (wvl - 443))
    return tf.cast(tsm_array * Anap * exp_term, tf.float32)

@tf.function
def bricaud_bp_model(tsm_array, Abp, Ebp, Bfp, wvl):
    tsm_array = tf.expand_dims(tsm_array, axis=-1)
    return tf.cast(Bfp * tsm_array * Abp * (wvl / 555.0) ** Ebp, tf.float32)

@tf.function
def log_bricaud_anap_model(tsm_array, Anap, Snap, wvl):
    tsm_original = tf.exp(tsm_array)
    tsm_original = tf.expand_dims(tsm_original, axis=-1)
    exp_term = tf.exp(-Snap * (wvl - 443))
    return tf.cast(tsm_original * Anap * exp_term, tf.float32)


@tf.function
def log_bricaud_bp_model(tsm_array, Abp, Ebp, Bfp, wvl):
    tsm_original = tf.exp(tsm_array)
    tsm_original = tf.expand_dims(tsm_original, axis=-1)
    return tf.cast(Bfp * tsm_original * Abp * (wvl / 555.0) ** Ebp, tf.float32)
