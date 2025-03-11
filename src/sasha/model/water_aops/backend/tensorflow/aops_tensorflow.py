import tensorflow as tf

@tf.function
def compute_kb(u, k, theta_v, theta_s, gamma_b, alpha_b, beta_b):
    gamma_b =tf.cast(gamma_b, dtype = tf.float32)
    alpha_b =tf.cast(alpha_b, dtype = tf.float32)
    beta_b  =tf.cast(beta_b, dtype = tf.float32)
    theta_v =tf.cast(theta_v, dtype = tf.float32)
    theta_s =tf.cast(theta_s, dtype = tf.float32)
    
    Dub = gamma_b * tf.sqrt(alpha_b + beta_b * u) / tf.cos(theta_v)
    return (1.0 / tf.cos(theta_s) + Dub/tf.cos(theta_v)) * k

@tf.function
def compute_rrsdp(u, alpha_dp, beta_dp):
    alpha_dp =tf.cast(alpha_dp, dtype = tf.float32)
    beta_dp =tf.cast(beta_dp, dtype = tf.float32)
    return (alpha_dp + beta_dp * u) * u