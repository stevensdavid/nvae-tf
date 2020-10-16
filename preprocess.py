from common import BNSwishConv
from config import SCALE_FACTOR
import tensorflow as tf
from tensorflow.keras import activations, Sequential, layers


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
            self.pre_process.add(BNSwishConv(2, n_channels, stride=(2, 2)))
            mult *= SCALE_FACTOR
        self.mult = mult

    def call(self, input):
        return self.pre_process(input)