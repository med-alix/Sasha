import tensorflow as tf

@tf.function
def compute_a(aphy, acdom, anap, awater):
    return aphy + acdom + anap + awater

@tf.function
def compute_bb(b_chl, b_cdom, b_tsm, b_pure):
    return b_tsm + b_pure + b_chl + b_cdom

@tf.function
def compute_k(a, bb):
    return a + bb

@tf.function
def compute_u(bb, k):
    return bb / k