from enum import Enum, auto
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import activations

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