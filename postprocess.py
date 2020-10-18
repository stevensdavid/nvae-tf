from common import RescaleType, Rescaler, SqueezeExcitation
from tensorflow.keras import activations
from tensorflow.keras import layers, Sequential
import tensorflow as tf


class Postprocess(layers.Layer):
    def __init__(
        self,
        n_blocks,
        n_cells,
        mult,
        n_channels_decoder,
        scale_factor,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.layers = Sequential()
        for block in range(n_blocks):
            # First cell rescales
            mult /= scale_factor
            output_channels = n_channels_decoder * mult
            for cell_idx in range(n_cells):
                self.layers.add(PostprocessCell(output_channels, n_nodes=1, upscale=cell_idx == 0, scale_factor=scale_factor))
        self.layers.add(layers.Conv2D(1, kernel_size=(3,3), padding="same"))
        self.mult = mult
        

    def call(self, input):
        return self.layers(input)


class PostprocessCell(layers.Layer):
    def __init__(self, n_channels, n_nodes, scale_factor, upscale=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layers = Sequential()
        if upscale:
            self.skip = Rescaler(n_channels, scale_factor=scale_factor, rescale_type=RescaleType.UP)
        else:
            self.skip = tf.identity
        for _ in range(n_nodes):
            self.layers.add(PostprocessNode(n_channels, upscale=upscale, scale_factor=scale_factor))
            if upscale:
                # Only scale once in each cells
                upscale = False

    def call(self, inputs):
        return self.skip(inputs) + self.layers(inputs)


class PostprocessNode(layers.Layer):
    def __init__(self, n_channels, upscale=False, expansion_ratio=6, scale_factor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layers = Sequential()
        if upscale:
            self.layers.add(Rescaler(n_channels, scale_factor, rescale_type=RescaleType.UP))
        self.layers.add(layers.BatchNormalization())
        hidden_dim = n_channels * expansion_ratio
        self.layers.add(ConvBNSwish(hidden_dim, kernel_size=(1,1), stride=(1,1)))
        self.layers.add(ConvBNSwish(hidden_dim, kernel_size=(5,5), stride=(1,1))) #, groups=int(hidden_dim)))
        self.layers.add(layers.Conv2D(n_channels, kernel_size=(1,1), strides=(1,1), use_bias=False))
        self.layers.add(layers.BatchNormalization())
        self.layers.add(SqueezeExcitation())

    def call(self, inputs):
        return self.layers(inputs)


class ConvBNSwish(layers.Layer):
    def __init__(self, n_channels, kernel_size, stride, groups=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layers = Sequential()
        self.layers.add(layers.Conv2D(n_channels, kernel_size=kernel_size, strides=stride, groups=groups, use_bias=False, padding="same"))
        self.layers.add(layers.BatchNormalization())
        self.layers.add(layers.Activation(activations.swish))

    def call(self, inputs):
        return self.layers(inputs)
