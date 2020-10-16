from typing import Sequence
from config import SCALE_FACTOR
from enum import Enum, auto
import tensorflow as tf

from tensorflow.keras import layers, activations, Sequential


class Sampler(layers.Layer):
    def __init__(self, n_latent_scales, n_groups_per_scale, n_latent_per_group, **kwargs) -> None:
        super().__init__(**kwargs)
        # Initialize sampler
        self.enc_sampler = []
        self.dec_sampler = []
        self.n_latent_scales = n_latent_scales
        self.n_groups_per_scale = n_groups_per_scale
        self.n_latent_per_group = n_latent_per_group
        for scale in range(self.n_latent_scales):
            n_groups = self.n_groups_per_scale[scale]
            for group in range(n_groups):
                self.enc_sampler.append(
                    # NVLabs use padding 1 here?
                    layers.Conv2D(
                        SCALE_FACTOR * self.n_latent_per_group, kernel_size=(3, 3), padding="same"
                    )
                )
                if scale == 0 and group == 0:
                    # Dummy value to maintain indexing
                    pass
                    # self.dec_sampler.append(None)
                else:
                    sampler = Sequential()
                    sampler.add(layers.ELU())
                    # NVLabs use padding 0 here?
                    sampler.add(layers.Conv2D(SCALE_FACTOR * self.n_latent_per_group, kernel_size=(1,1)))
                    self.dec_sampler.append(sampler)


    def sample(self, mu, log_sigma):
        # reparametrization trick
        z = mu + tf.random.normal(shape=mu.shape) * tf.math.exp(log_sigma)
        return z

    def get_params(self, sampler, z_idx, prior):
        params = sampler[z_idx](prior)
        mu, log_sigma = tf.split(params, 2, axis=-1)
        mu, log_sigma = [tf.squeeze(p) for p in (mu, log_sigma)]
        return mu, log_sigma

    def call(self, prior, z_idx):
        enc_mu, enc_log_sigma = self.get_params(self.enc_sampler, z_idx, prior)
        if z_idx == 0:
            return self.sample(enc_mu, enc_log_sigma)
        # Get decoder offsets
        dec_mu, dec_log_sigma = self.get_params(self.dec_sampler, z_idx, prior)
        return self.sample(enc_mu + dec_mu, enc_log_sigma + dec_log_sigma)


class RescaleType(Enum):
    UP = auto()
    DOWN = auto()


class SqueezeExcitation(layers.Layer):
    """Squeeze and Excitation block as defined by Hu, et al. (2019)

    See Also
    ========
    Source paper https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, ratio=16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        batch_size, h, w, c = input_shape
        self.gap = layers.GlobalAveragePooling2D(data_format="channels_last")
        num_hidden = max(c / self.ratio, 4)
        self.dense1 = layers.Dense(units=num_hidden)
        self.dense2 = layers.Dense(units=c)

    def call(self, input):
        # x = tf.math.reduce_mean(x, axis=[1, 2])
        x = self.gap(input)
        x = self.dense1(x)
        x = activations.relu(x)
        x = self.dense2(x)
        x = activations.sigmoid(x)
        x = tf.reshape(x, (x.shape[0], 1, 1, -1))
        return x * input

class Rescaler(layers.Layer):
    def __init__(self, n_channels, scale_factor, rescale_type, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bn = layers.BatchNormalization()
        self.mode = rescale_type
        self.factor = scale_factor
        if rescale_type == RescaleType.UP:
            self.conv = layers.Conv2D(n_channels, (3, 3), strides=(1, 1), padding="same")
        elif rescale_type == RescaleType.DOWN:
            self.conv = layers.Conv2D(
                n_channels, (3, 3), strides=(self.factor, self.factor), padding="same"
            )

    def call(self, input):
        x = self.bn(input)
        x = activations.swish(x)
        if self.mode == RescaleType.UP:
            _, height, width, _ = x.shape
            x = tf.image.resize(
                x, size=(self.factor * height, self.factor * width), method="nearest"
            )
        x = self.conv(x)
        return x