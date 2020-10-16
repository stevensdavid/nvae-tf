from typing import Tuple
from config import SCALE_FACTOR
from enum import Enum, auto
import tensorflow as tf

from tensorflow.keras import layers, activations, Sequential
from dataclasses import dataclass


@dataclass
class DistributionParams:
    enc_mu: float
    enc_log_sigma: float
    dec_mu: float
    dec_log_sigma: float


class SkipScaler(layers.Layer):
    def __init__(self, n_channels, **kwargs):
        super().__init__(**kwargs)
        # Each convolution handles a quarter of the channels
        self.conv1 = layers.Conv2D(n_channels // 4, (1,1), strides=(2,2), padding="same")
        self.conv2 = layers.Conv2D(n_channels // 4, (1,1), strides=(2,2), padding="same")
        self.conv3 = layers.Conv2D(n_channels // 4, (1,1), strides=(2,2), padding="same")
        # This convolotuion handles the remaining channels
        self.conv4 = layers.Conv2D(n_channels - 3 * (n_channels // 4), (1,1), strides=(2,2), padding="same")

    def call(self, x):
        out = activations.swish(x)
        # Indexes are offset as we stride by 2x2, this way we cover all pixels
        conv1 = self.conv1(out)
        conv2 = self.conv2(out[:, 1:, 1:, :])
        conv3 = self.conv3(out[:, :, 1:, :])
        conv4 = self.conv4(out[:, 1:, :, :])
        # Combine channels
        out = tf.concat((conv1, conv2, conv3, conv4), axis=3)
        return out

class BNSwishConv(layers.Layer):
    def __init__(self, n_nodes, n_channels, stride, **kwargs) -> None:
        super().__init__(**kwargs)
        self.nodes = Sequential()
        if stride == (1,1):
            self.skip = tf.identity
        elif stride == (2,2):
            # We have to rescale the input in order to combine it
            self.skip = SkipScaler(n_channels)
        for i in range(n_nodes):
            self.nodes.add(layers.BatchNormalization())
            self.nodes.add(layers.Activation(activations.swish))
            # 
            self.nodes.add(layers.Conv2D(
                n_channels, 
                (3, 3), 
                # Only apply rescaling on first node
                stride if i == 0 else (1,1), 
                padding="same"
            ))
        self.se = SqueezeExcitation()

    def call(self, input):
        skipped = self.skip(input)
        x = self.nodes(input)
        x = self.se(x)
        return skipped + x

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

    def call(self, prior, z_idx) -> Tuple[tf.Tensor, DistributionParams]:
        enc_mu, enc_log_sigma = self.get_params(self.enc_sampler, z_idx, prior)
        if z_idx == 0:
            params = DistributionParams(enc_mu, enc_log_sigma, None, None)
            return self.sample(enc_mu, enc_log_sigma), params
        # Get decoder offsets
        dec_mu, dec_log_sigma = self.get_params(self.dec_sampler, z_idx, prior)
        params = DistributionParams(enc_mu, enc_log_sigma, dec_mu, dec_log_sigma)
        z = self.sample(enc_mu + dec_mu, enc_log_sigma + dec_log_sigma)
        return z, params


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