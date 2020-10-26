from common import SqueezeExcitation
import tensorflow as tf
from tensorflow.keras import activations, Sequential, layers
from tensorflow_addons.layers import SpectralNormalization


class Preprocess(tf.keras.Model):
    def __init__(
        self, n_encoder_channels, n_blocks, n_cells, scale_factor, input_shape, mult=1, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.pre_process = Sequential(
            SpectralNormalization(
                layers.Conv2D(n_encoder_channels, (3, 3), padding="same")
            )
        )
        for block in range(n_blocks):
            for cell in range(n_cells - 1):
                n_channels = mult * n_encoder_channels
                cell = BNSwishConv(2, n_channels, stride=(1, 1))
                self.pre_process.add(cell)
            # Rescale channels on final cell
            n_channels = mult * n_encoder_channels * scale_factor
            self.pre_process.add(BNSwishConv(2, n_channels, stride=(2, 2)))
            mult *= scale_factor
            input_shape *= [1, 1/scale_factor, 1/scale_factor, scale_factor]
        self.mult = mult
        self.output_shape_ = input_shape

    def call(self, inputs):
        return self.pre_process(2 * inputs - 1)


class SkipScaler(tf.keras.Model):
    def __init__(self, n_channels, **kwargs):
        super().__init__(**kwargs)
        # Each convolution handles a quarter of the channels
        self.conv1 = SpectralNormalization(
            layers.Conv2D(n_channels // 4, (1, 1), strides=(2, 2), padding="same")
        )
        self.conv2 = SpectralNormalization(
            layers.Conv2D(n_channels // 4, (1, 1), strides=(2, 2), padding="same")
        )
        self.conv3 = SpectralNormalization(
            layers.Conv2D(n_channels // 4, (1, 1), strides=(2, 2), padding="same")
        )
        # This convolotuion handles the remaining channels
        self.conv4 = SpectralNormalization(
            layers.Conv2D(
                n_channels - 3 * (n_channels // 4),
                (1, 1),
                strides=(2, 2),
                padding="same",
            )
        )

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


class BNSwishConv(tf.keras.Model):
    def __init__(self, n_nodes, n_channels, stride, **kwargs) -> None:
        super().__init__(**kwargs)
        self.nodes = Sequential()
        if stride == (1, 1):
            self.skip = tf.identity
        elif stride == (2, 2):
            # We have to rescale the input in order to combine it
            self.skip = SkipScaler(n_channels)
        for i in range(n_nodes):
            self.nodes.add(layers.BatchNormalization(momentum=0.05))
            self.nodes.add(layers.Activation(activations.swish))
            #
            self.nodes.add(
                SpectralNormalization(
                    layers.Conv2D(
                        n_channels,
                        (3, 3),
                        # Only apply rescaling on first node
                        stride if i == 0 else (1, 1),
                        padding="same",
                    )
                )
            )
        self.se = SqueezeExcitation()

    def call(self, inputs):
        skipped = self.skip(inputs)
        x = self.nodes(inputs)
        x = self.se(x)
        return skipped + x
