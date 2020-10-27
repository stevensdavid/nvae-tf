# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

# Taken from NVIDIA at https://github.com/NVlabs/stylegan2/blob/master/metrics/perceptual_path_length.py

"""Perceptual Path Length (PPL)."""

import numpy as np
import tensorflow as tf


# ----------------------------------------------------------------------------

# Normalize batch of vectors.
def normalize(v):
    return v / tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1, keepdims=True))


# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = tf.reduce_sum(a * b, axis=-1, keepdims=True)
    p = tf.reshape(t, [-1, 1,1,1]) * tf.math.acos(d)
    c = normalize(b - d * a)
    d = a * tf.math.cos(p) + c * tf.math.sin(p)
    return normalize(d)


# ----------------------------------------------------------------------------


def evaluate(act1, act2, epsilon=1e-4):
    distances = tf.linalg.norm(act1 - act2, axis=0)
    distances = distances * (1 / epsilon ** 2)
    # Reject outliers.
    lo = np.percentile(distances, 1, interpolation="lower")
    hi = np.percentile(distances, 99, interpolation="higher")
    filtered_distances = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )
    return tf.reduce_mean(filtered_distances)


# ----------------------------------------------------------------------------
