import tensorflow as tf
import numpy as np

@tf.function
def compute_rrsw(z, kc, rrs_dp):
    z = tf.expand_dims(z, axis=-1)
    rrsc = rrs_dp * (1.0 - tf.exp(-kc * z))
    return rrsc


@tf.function
def compute_rrsb(z, kb, albedo):
    z = tf.expand_dims(z, axis=-1)
    rrsb = (albedo / tf.constant(np.pi, dtype=tf.float32)) * tf.exp(-kb * z)
    return rrsb

@tf.function
def compute_rrsm(z, kc, rrs_dp, kb, albedo):
    rrsw = compute_rrsw(z, kc, rrs_dp)
    rrsb = compute_rrsb(z, kb, albedo)
    return rrsw + rrsb

@tf.function
def compute_rrsp(z, kc, rrs_dp, kb, albedo, nconv, dconv):
    rrsm = compute_rrsm(z, kc, rrs_dp, kb, albedo)
    rrsp = nconv * rrsm / (1.0 - dconv * rrsm)
    return rrsp