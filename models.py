import tensorflow as tf
from tensorflow.keras import layers, activations, Sequential


class NVAE(tf.keras.Model):
    def __init__(self, n_encoder_channels, res_cells_per_group, n_groups, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(n_encoder_channels, res_cells_per_group, n_groups)
        self.decoder = Decoder()

    def call(self, x):
        x = self.encoder(x)
        return x

    def sample(self):
        pass


class Encoder(layers.Layer):
    def __init__(self, n_encoder_channels, res_cells_per_group, n_groups, **kwargs):
        super().__init__(**kwargs)
        self.pre_process = layers.Conv2D(n_encoder_channels, (3,3))
        self.groups = []
        for i in range(n_groups):
            output_channels = n_encoder_channels / (2**i)
            group = Sequential(
                layers.Conv2D(output_channels, (3,3))
            )
            for _ in range(res_cells_per_group):
                group.add(EncodingResidualCell(output_channels))
            self.groups.append(group)

    def call(self, x):
        x = self.pre_process(x)
        # 8x26x26x32
        outputs = [x]
        for group in self.groups:
            outputs.append(group(outputs[-1]))
        return outputs[1:]


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
        self.gap = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(units=c/self.ratio)
        self.dense2 = layers.Dense(units=c)

    def call(self, x):
        x = self.gap(x)
        x = self.dense1(x)
        x = activations.relu(x)
        x = self.dense2(x)
        return activations.sigmoid(x)


class GenerativeResidualCell(layers.Layer):
    """Generative network residual cell in NVAE architecture
    """
    def __init__(self, output_channels, expansion_ratio=6, **kwargs):
        super().__init__(**kwargs)
        self.batch_norm1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(expansion_ratio * output_channels, (1,1))
        self.batch_norm2 = layers.BatchNormalization()
        self.depth_conv = layers.DepthwiseConv2D((5,5))
        self.batch_norm3 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(output_channels, (1,1))
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
    def __init__(self, output_channels, **kwargs):
        super().__init__(**kwargs)
        self.batch_norm1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(output_channels, (3,3))
        self.batch_norm2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(output_channels, (3,3))
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
