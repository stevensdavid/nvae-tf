
from common import RescaleType, Rescaler, SqueezeExcitation
from typing import List
from tensorflow.keras import layers, Sequential, activations
from models import SCALE_FACTOR


class EncoderGroupActivation(layers.Layer):
    def __init__(self, n_channels, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(n_channels, (1, 1))

    def call(self, input):
        x = self.conv(input)
        return input + x


class Encoder(layers.Layer):
    def __init__(
        self,
        n_encoder_channels,
        n_latent_per_group: int,
        res_cells_per_group,
        n_latent_scales: int,
        n_groups_per_scale: List[int],
        mult: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Initialize encoder tower
        self.groups = []
        for scale in range(n_latent_scales):
            n_groups = n_groups_per_scale[scale]
            for group in range(n_groups):
                output_channels = n_encoder_channels * mult
                group = Sequential()
                for _ in range(res_cells_per_group):
                    group.add(EncodingResidualCell(output_channels))
                self.groups.append(group)
                if not (scale == n_latent_scales - 1 and group == n_groups - 1):
                    # We apply a convolutional between each group except the final output
                    self.groups.append(EncoderGroupActivation(output_channels))
            # We downsample in the end of each scale except last
            if scale < n_latent_scales - 1:
                output_channels = n_encoder_channels * mult
                self.groups.append(
                    Rescaler(
                        output_channels, scale_factor=2, rescale_type=RescaleType.DOWN
                    )
                )
                mult *= SCALE_FACTOR
        self.final_enc = Sequential(
            [
                layers.ELU(),
                layers.Conv2D(n_encoder_channels * mult, (1, 1), padding="same"),
                layers.ELU(),
            ]
        )
        # Initialize sampler
        self.sampler = []
        for scale in range(n_latent_scales):
            n_groups = n_groups_per_scale[scale]
            for group in range(n_groups):
                self.sampler.append(
                    layers.Conv2D(
                        SCALE_FACTOR * n_latent_per_group, (3, 3), padding="same"
                    )
                )
        self.mult = mult

    def call(self, x):
        x = self.pre_process(x)
        # 8x26x26x32
        group_outputs = []
        combiners = []
        for group in self.groups:
            x = group(x)
            if isinstance(group, EncoderGroupActivation):
                # We are stepping between groups, need to save results
                group_outputs.append(x)
                combiners.append(group)
        final = self.final_enc(x)
        return group_outputs, combiners, final


class EncodingResidualCell(layers.Layer):
    """Encoding network residual cell in NVAE architecture"""

    def __init__(self, output_channels, **kwargs):
        super().__init__(**kwargs)
        self.batch_norm1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(output_channels, (3, 3), padding="same")
        self.batch_norm2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(output_channels, (3, 3), padding="same")
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
