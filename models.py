from typing import List
import tensorflow as tf
from tensorflow.keras import layers, activations, Sequential
from enum import Enum, auto
from tensorflow.python.keras.layers.convolutional import Conv2D

from tensorflow.python.keras.layers.normalization import BatchNormalization


class RescaleType(Enum):
    UP = auto()
    DOWN = auto()


class NVAE(tf.keras.Model):
    def __init__(self, n_encoder_channels, res_cells_per_group, n_groups, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(
            n_encoder_channels=3,
            n_latent_per_group=1,
            res_cells_per_group=2,
            n_latent_scales=2,
            n_groups_per_scale=[3, 1],
        )
        self.decoder = Decoder()

    def call(self, x):
        group_outputs, combiners, final_x = self.encoder(x)
        dist_params = self.encoder.sampler[0](final_x)
        mu, log_sigma = tf.split(dist_params, 2, axis=-1)
        mu = tf.squeeze(mu)
        log_sigma = tf.squeeze(log_sigma)
        z = tf.random.normal(shape=mu.shape, mean=mu, stddev=tf.math.exp(log_sigma))
        return x

    def sample(self):
        pass


class EncoderCombiner(layers.Layer):
    def __init__(self, n_channels, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(n_channels, (1, 1))

    def call(self, input):
        x = self.conv(input)
        return input + x


class Rescaler(layers.Layer):
    def __init__(self, n_channels, scale_factor, rescale_type, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bn = BatchNormalization()
        self.mode = rescale_type
        self.factor = scale_factor
        if rescale_type == RescaleType.UP:
            self.conv = Conv2D(n_channels, (3, 3), strides=(1, 1), padding="same")
        elif rescale_type == RescaleType.DOWN:
            self.conv = Conv2D(
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


class Encoder(layers.Layer):
    def __init__(
        self,
        n_encoder_channels,
        n_latent_per_group: int,
        res_cells_per_group,
        n_latent_scales: int,
        n_groups_per_scale: List[int],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pre_process = layers.Conv2D(n_encoder_channels, (3, 3), padding="same")

        # Initialize encoder tower
        self.groups = []
        for scale in range(n_latent_scales):
            n_groups = n_groups_per_scale[scale]
            for group in range(n_groups):
                output_channels = n_encoder_channels * (2 ** scale)
                group = Sequential()
                for _ in range(res_cells_per_group):
                    group.add(EncodingResidualCell(output_channels))
                self.groups.append(group)
                if not (scale == n_latent_scales - 1 and group == n_groups - 1):
                    # We apply a convolutional between each group except the final output
                    self.groups.append(EncoderCombiner(output_channels))
            # We downsample in the end of each scale except last
            if scale < n_latent_scales - 1:
                output_channels = 2 * n_encoder_channels
                self.groups.append(
                    Rescaler(
                        output_channels, scale_factor=2, rescale_type=RescaleType.DOWN
                    )
                )
        self.final_enc = Sequential(
            [
                layers.ELU(),
                layers.Conv2D(n_encoder_channels, (1, 1), padding="same"),
                layers.ELU(),
            ]
        )
        # Initialize sampler
        self.sampler = []
        for scale in range(n_latent_scales):
            n_groups = n_groups_per_scale[scale]
            for group in range(n_groups):
                self.sampler.append(
                    layers.Conv2D(2 * n_latent_per_group, (3, 3), padding="same")
                )

    def call(self, x):
        x = self.pre_process(x)
        # 8x26x26x32
        group_outputs = []
        combiners = []
        for group in self.groups:
            x = group(x)
            if isinstance(group, EncoderCombiner):
                # We are stepping between groups, need to save results
                group_outputs.append(x)
                combiners.append(group)
        final = self.final_enc(x)
        return group_outputs, combiners, final


class Decoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x


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


class GenerativeResidualCell(layers.Layer):
    """Generative network residual cell in NVAE architecture"""

    def __init__(self, output_channels, expansion_ratio=6, **kwargs):
        super().__init__(**kwargs)
        self.batch_norm1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(
            expansion_ratio * output_channels, (1, 1), padding="same"
        )
        self.batch_norm2 = layers.BatchNormalization()
        self.depth_conv = layers.DepthwiseConv2D((5, 5), padding="same")
        self.batch_norm3 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(output_channels, (1, 1), padding="same")
        self.batch_norm4 = layers.BatchNormalization()
        self.se = SqueezeExcitation()

    def call(self, input):
        x = self.batch_norm1(input)
        x = self.conv1(x)
        x = activations.swish(self.batch_norm2(x))
        x = self.depth_conv(x)
        x = activations.swish(self.batch_norm3(x))
        x = self.conv2(x)
        x = self.batch_norm4(x)
        x = self.se(x)
        return input + x


class EncodingResidualCell(layers.Layer):
    """Encoding network residual cell in NVAE architecture"""

    def __init__(self, output_channels, **kwargs):
        super().__init__(**kwargs)
        self.batch_norm1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(output_channels, (3, 3), padding="same")
        self.batch_norm2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(output_channels, (3, 3), padding="same")
        self.se = SqueezeExcitation()

    def call(self, input):
        # 8x24x24x32
        x = activations.swish(self.batch_norm1(input))
        # 8x24x24x32
        x = self.conv1(x)
        # 8x22x22x32
        x = activations.swish(self.batch_norm2(x))
        # 8x22x22x32
        x = self.conv2(x)
        # 8x20x20x32
        x = self.se(x)
        # 8x32
        return input + x
