import tensorflow as tf
from opt_einsum import contract

@tf.function
def bricaud_cdom_model(cdom_array, S_cdom, wvl):
    cdom_array = tf.expand_dims(cdom_array, axis=-1)
    exp_term = tf.exp(-S_cdom * (wvl - 443))
    return tf.cast(cdom_array * exp_term, tf.float32)

@tf.function
def log_bricaud_cdom_model(cdom_array, S_cdom, wvl):
    cdom_original = tf.exp(cdom_array)
    cdom_original = tf.expand_dims(cdom_original, axis=-1)
    exp_term      = tf.exp(-S_cdom * (wvl - 443))
    return tf.cast(cdom_original * exp_term, tf.float32)



# @tf.function
# def log_bricaud_cdom_model(cdom_array, S_cdom, wvl):
#     cdom_original = tf.exp(cdom_array)
#     exp_term = tf.exp(-S_cdom * (wvl - 443))
#     return contract('i,j->ij', cdom_original, exp_term)
#     0532584859
#     0749277129