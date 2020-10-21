from common import RescaleType, Rescaler, SqueezeExcitation
from typing import List
from tensorflow.keras import layers, Sequential, activations
from functools import partial
import tensorflow as tf


class EncoderDecoderCombiner(tf.keras.Model):
    def __init__(self, n_channels, **kwargs) -> None:
        super().__init__(**kwargs)
        self.decoder_conv = layers.Conv2D(n_channels, (1, 1))

    def call(self, encoder_x, decoder_x):
        x = self.decoder_conv(decoder_x)
        return encoder_x + x


class Encoder(tf.keras.Model):
    def __init__(
        self,
        n_encoder_channels,
        n_latent_per_group: int,
        res_cells_per_group,
        n_latent_scales: int,
        n_groups_per_scale: List[int],
        mult: int,
        scale_factor: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Initialize encoder tower
        self.groups = []
        for scale in range(n_latent_scales):
            n_groups = n_groups_per_scale[scale]
            for group_idx in range(n_groups):
                output_channels = n_encoder_channels * mult
                group = Sequential()
                for _ in range(res_cells_per_group):
                    group.add(EncodingResidualCell(output_channels))
                self.groups.append(group)
                if not (scale == n_latent_scales - 1 and group_idx == n_groups - 1):
                    # We apply a convolutional between each group except the final output
                    self.groups.append(EncoderDecoderCombiner(output_channels))
            # We downsample in the end of each scale except last
            if scale < n_latent_scales - 1:
                output_channels = n_encoder_channels * mult * scale_factor
                self.groups.append(
                    Rescaler(
                        output_channels,
                        scale_factor=scale_factor,
                        rescale_type=RescaleType.DOWN,
                    )
                )
                mult *= scale_factor
        self.final_enc = Sequential(
            [
                layers.ELU(),
                layers.Conv2D(n_encoder_channels * mult, (1, 1), padding="same"),
                layers.ELU(),
            ]
        )
        self.mult = mult

    def call(self, x):
        # 8x26x26x32
        enc_dec_combiners = []
        for group in self.groups:
            if isinstance(group, EncoderDecoderCombiner):
                # We are stepping between groups, need to save results
                enc_dec_combiners.append(
                    partial(group, x)
                    # lambda dec_x, enc_x=x, combiner=group: combiner(enc_x, dec_x)
                )
            else:
                x = group(x)
        final = self.final_enc(x)
        return enc_dec_combiners, final


class EncodingResidualCell(tf.keras.Model):
    """Encoding network residual cell in NVAE architecture"""

    def __init__(self, output_channels, **kwargs):
        super().__init__(**kwargs)
        self.batch_norm1 = layers.BatchNormalization(momentum=.05)
        self.conv1 = layers.Conv2D(output_channels, (3, 3), padding="same")
        self.batch_norm2 = layers.BatchNormalization(momentum=.05)
        self.conv2 = layers.Conv2D(output_channels, (3, 3), padding="same")
        self.se = SqueezeExcitation()

    def call(self, inputs):
        # 8x24x24x32
        x = activations.swish(self.batch_norm1(inputs))
        # 8x24x24x32
        x = self.conv1(x)
        # 8x22x22x32
        x = activations.swish(self.batch_norm2(x))
        # 8x22x22x32
        x = self.conv2(x)
        # 8x20x20x32
        x = self.se(x)
        # 8x32
        return inputs + x
