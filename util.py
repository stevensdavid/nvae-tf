import tensorflow as tf


def tile_images(images):
    n_images = tf.cast(tf.shape(images)[0], float)
    # Convert to side of square
    n = int(tf.math.floor(tf.math.sqrt(n_images)))
    _, height, width, channels = tf.shape(images)
    images = tf.reshape(images, [n, n, height, width, channels])
    images = tf.transpose(images, perm=[2, 0, 3, 1, 4])
    return tf.reshape(images, [n * height, n * width, channels])
