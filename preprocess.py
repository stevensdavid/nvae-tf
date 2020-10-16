from models import SCALE_FACTOR
from common import SqueezeExcitation
import tensorflow as tf
from tensorflow.keras import activations, Sequential, layers


class PreprocessSkipScaler(layers.Layer):
    def __init__(self, n_channels, **kwargs):
        super().__init__(**kwargs)
        # Each convolution handles a quarter of the channels
        self.conv1 = layers.Conv2D(n_channels // 4, (1,1), strides=(2,2))
        self.conv2 = layers.Conv2D(n_channels // 4, (1,1), strides=(2,2))
        self.conv3 = layers.Conv2D(n_channels // 4, (1,1), strides=(2,2))
        # This convolotuion handles the remaining channels
        self.conv4 = layers.Conv2D(n_channels - 3 * (n_channels // 4), (1,1), strides=(2,2))

    def forward(self, x):
        out = activations.swish(x)
        conv1 = self.conv_1(out)
        conv2 = self.conv_2(out[:, :, 1:, 1:])
        conv3 = self.conv_3(out[:, :, :, 1:])
        conv4 = self.conv_4(out[:, :, 1:, :])
        out = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        return out


class BNSwishConv(layers.Layer):
    def __init__(self, n_nodes, n_channels, stride, **kwargs) -> None:
        super().__init__(**kwargs)
        self.nodes = Sequential()
        if stride == (1,1):
            self.skip = tf.identity
        elif stride == (2,2):
            self.skip = ...
        for _ in range(n_nodes):
            self.nodes.add(layers.BatchNormalization())
            self.nodes.add(layers.Activation(activations.swish))
            self.nodes.add(layers.Conv2D(n_channels, (3, 3), stride, padding="same"))
        self.se = SqueezeExcitation()

    def call(self, input):
        x = self.nodes(input)
        x = self.se(x)
        return input + x


class Preprocess(layers.Layer):
    def __init__(self, n_encoder_channels, n_blocks, n_cells, mult=1, **kwargs) -> None:
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
            n_channels = mult * n_encoder_channels * SCALE_FACTOR
            # TODO: NVLabs uses FactorizedReduce here as skip connection...
            self.pre_process.add(BNSwishConv(2, n_channels, stride=(2, 2)))
            mult *= SCALE_FACTOR
        self.mult = mult

    def call(self, input):
        return self.pre_process(input)