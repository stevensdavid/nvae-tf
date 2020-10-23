from common import RescaleType, Rescaler, SqueezeExcitation
from tensorflow.keras import activations
from tensorflow.keras import layers, Sequential
import tensorflow as tf
from tensorflow_addons.layers import SpectralNormalization


class Postprocess(tf.keras.Model):
    def __init__(
        self, n_blocks, n_cells, mult, n_channels_decoder, scale_factor, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.sequence = Sequential()
        for block in range(n_blocks):
            # First cell rescales
            mult /= scale_factor
            output_channels = n_channels_decoder * mult
            for cell_idx in range(n_cells):
                self.sequence.add(
                    PostprocessCell(
                        output_channels,
                        n_nodes=1,
                        upscale=cell_idx == 0,
                        scale_factor=scale_factor,
                    )
                )
        self.sequence.add(layers.Activation(activations.elu))
        self.sequence.add(
            SpectralNormalization(layers.Conv2D(1, kernel_size=(3, 3), padding="same"))
        )
        self.mult = mult

    def call(self, inputs):
        return self.sequence(inputs)


class PostprocessCell(tf.keras.Model):
    def __init__(
        self, n_channels, n_nodes, scale_factor, upscale=False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.sequence = Sequential()
        if upscale:
            self.skip = Rescaler(
                n_channels, scale_factor=scale_factor, rescale_type=RescaleType.UP
            )
        else:
            self.skip = tf.identity
        for _ in range(n_nodes):
            self.sequence.add(
                PostprocessNode(n_channels, upscale=upscale, scale_factor=scale_factor)
            )
            if upscale:
                # Only scale once in each cells
                upscale = False

    def call(self, inputs):
        return self.skip(inputs) + self.sequence(inputs)


class PostprocessNode(tf.keras.Model):
    def __init__(
        self, n_channels, scale_factor, upscale=False, expansion_ratio=6, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.sequence = Sequential()
        if upscale:
            self.sequence.add(
                Rescaler(n_channels, scale_factor, rescale_type=RescaleType.UP)
            )
        self.sequence.add(layers.BatchNormalization(momentum=0.05))
        hidden_dim = n_channels * expansion_ratio
        self.sequence.add(ConvBNSwish(hidden_dim, kernel_size=(1, 1), stride=(1, 1)))
        self.sequence.add(
            ConvBNSwish(hidden_dim, kernel_size=(5, 5), stride=(1, 1))
        )  # , groups=int(hidden_dim)))
        self.sequence.add(
            SpectralNormalization(
                layers.Conv2D(
                    n_channels, kernel_size=(1, 1), strides=(1, 1), use_bias=False
                )
            )
        )
        self.sequence.add(layers.BatchNormalization(momentum=0.05))
        self.sequence.add(SqueezeExcitation())

    def call(self, inputs):
        return self.sequence(inputs)


class ConvBNSwish(tf.keras.Model):
    def __init__(self, n_channels, kernel_size, stride, groups=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sequence = Sequential()
        self.sequence.add(
            SpectralNormalization(
                layers.Conv2D(
                    n_channels,
                    kernel_size=kernel_size,
                    strides=stride,
                    groups=groups,
                    use_bias=False,
                    padding="same",
                )
            )
        )
        self.sequence.add(layers.BatchNormalization(momentum=0.05))
        self.sequence.add(layers.Activation(activations.swish))

    def call(self, inputs):
        return self.sequence(inputs)
