from common import SqueezeExcitation
import tensorflow as tf
from tensorflow.keras import activations, Sequential, layers


class Preprocess(layers.Layer):
    def __init__(
        self, n_encoder_channels, n_blocks, n_cells, scale_factor, mult=1, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.pre_process = Sequential(
            layers.Conv2D(n_encoder_channels, (3, 3), padding="same")
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
        self.mult = mult

    def call(self, input):
        return self.pre_process(input)


class SkipScaler(layers.Layer):
    def __init__(self, n_channels, **kwargs):
        super().__init__(**kwargs)
        # Each convolution handles a quarter of the channels
        self.conv1 = layers.Conv2D(
            n_channels // 4, (1, 1), strides=(2, 2), padding="same"
        )
        self.conv2 = layers.Conv2D(
            n_channels // 4, (1, 1), strides=(2, 2), padding="same"
        )
        self.conv3 = layers.Conv2D(
            n_channels // 4, (1, 1), strides=(2, 2), padding="same"
        )
        # This convolotuion handles the remaining channels
        self.conv4 = layers.Conv2D(
            n_channels - 3 * (n_channels // 4), (1, 1), strides=(2, 2), padding="same"
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


class BNSwishConv(layers.Layer):
    def __init__(self, n_nodes, n_channels, stride, **kwargs) -> None:
        super().__init__(**kwargs)
        self.nodes = Sequential()
        if stride == (1, 1):
            self.skip = tf.identity
        elif stride == (2, 2):
            # We have to rescale the input in order to combine it
            self.skip = SkipScaler(n_channels)
        for i in range(n_nodes):
            self.nodes.add(layers.BatchNormalization())
            self.nodes.add(layers.Activation(activations.swish))
            #
            self.nodes.add(
                layers.Conv2D(
                    n_channels,
                    (3, 3),
                    # Only apply rescaling on first node
                    stride if i == 0 else (1, 1),
                    padding="same",
                )
            )
        self.se = SqueezeExcitation()

    def call(self, input):
        skipped = self.skip(input)
        x = self.nodes(input)
        x = self.se(x)
        return skipped + x
