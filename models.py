import tensorflow as tf
from tensorflow.keras import layers, activations


class NVAE(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, x):
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
        h, w, c = input_shape
        self.gap = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(units=c/self.ratio)
        self.dense2 = layers.Dense(unts=c)

    def call(self, x):
        x = self.gap(x)
        x = self.dense1(x)
        x = activations.relu(x)
        x = self.dense2(x)
        return activations.sigmoid(x)


class GenerativeResidualCell(layers.Layer):
    """Generative network residual cell in NVAE architecture
    """
    def __init__(self, n_filters, **kwargs):
        super().__init__(**kwargs)
        self.batch_norm1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(n_filters, (1,1))
        self.batch_norm2 = layers.BatchNormalization()
        self.depth_conv = layers.DepthwiseConv2D((5,5))
        self.batch_norm3 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(n_filters, (1,1))
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
    """Encoding network residual cell in NVAE architecture
    """
    def __init__(self, n_filters, **kwargs):
        super().__init__(**kwargs)
        self.batch_norm1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(n_filters, (3,3))
        self.batch_norm2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(n_filters, (3,3))
        self.se = SqueezeExcitation()

    def call(self, input):
        x = activations.swish(self.batch_norm1(input))
        x = self.conv1(x)
        x = activations.swish(self.batch_norm2(x))
        x = self.conv2(x)
        x = self.se(x)
        return input + x
