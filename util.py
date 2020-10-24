import tensorflow as tf
import math


def tile_images(images):
    n_images = tf.cast(tf.shape(images)[0], float)
    # Convert to side of square
    n = int(tf.math.floor(tf.math.sqrt(n_images)))
    _, height, width, channels = tf.shape(images)
    images = tf.reshape(images, [n, n, height, width, channels])
    images = tf.transpose(images, perm=[2, 0, 3, 1, 4])
    return tf.reshape(images, [n * height, n * width, channels])


def calculate_log_p(z, mu, log_sigma):
    sigma = tf.math.exp(log_sigma) + 1e-2
    normalized_z = (z - mu) / sigma
    log_p = -0.5 * normalized_z * normalized_z - 0.5 * tf.math.log(2*tf.constant(math.pi)) - log_sigma
    return log_p
