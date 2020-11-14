from dataclasses import dataclass
import io
import os
import uuid
import numpy as np
from typing import Any, List
import tensorflow as tf
from tqdm.std import trange
import math


def tile_images(images):
    n_images = tf.cast(tf.shape(images)[0], float)
    # Convert to side of square
    n = int(tf.math.floor(tf.math.sqrt(n_images)))
    _, height, width, channels = tf.shape(images)
    images = tf.reshape(images, [n, n, height, width, channels])
    images = tf.transpose(images, perm=[2, 0, 3, 1, 4])
    return tf.reshape(images, [n * height, n * width, channels])


def sample_to_dir(model, batch_size, sample_size, temperature, output_dir):
    batches = max(sample_size // batch_size, 1)
    for image_batch in trange(batches, desc="Generating samples"):
        images, *_ = model.sample(
            n_samples=batch_size, return_mean=False, temperature=temperature
        )
        save_images_to_dir(images, output_dir)


def save_images_to_dir(images, dir):
    if images.dtype.is_floating:
        images = tf.cast(images * 255, tf.uint8)
    for image in images:
        encoded = tf.io.encode_png(image)
        tf.io.write_file(os.path.join(dir, f"{uuid.uuid4()}.png"), encoded)


def calculate_log_p(z, mu, sigma):
    normalized_z = (z - mu) / sigma
    log_p = (
        -0.5 * normalized_z * normalized_z
        - 0.5 * tf.math.log(2 * tf.constant(math.pi))
        - tf.math.log(sigma)
    )
    return log_p


def softclamp5(x):
    return 5.0 * tf.math.tanh(x / 5.0)  # differentiable clamp [-5, 5]


@dataclass
class Metric:
    mean: float
    stddev: float

    @staticmethod
    def from_list(l):
        return Metric(mean=np.mean(l), stddev=np.std(l))


@dataclass
class Metrics:
    temperature: float
    fid: float
    ppl: Metric
    precision: Metric
    recall: Metric


@dataclass
class ModelEvaluation:
    nll: Metric
    sample_metrics: List[Metrics]
