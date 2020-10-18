from typing import List
from common import RescaleType, SqueezeExcitation, Rescaler, Sampler
import tensorflow as tf
from tensorflow.keras import layers, activations, Sequential


class Decoder(layers.Layer):
    def __init__(
        self,
        n_decoder_channels,
        n_latent_per_group: int,
        res_cells_per_group,
        n_latent_scales: int,
        n_groups_per_scale: List[int],
        mult: int,
        scale_factor: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        # these 4s should be changed
        self.h = tf.Variable(tf.zeros((4, 4, n_decoder_channels)), trainable=True)
        self.sampler = Sampler(n_latent_scales=n_latent_scales, n_groups_per_scale=n_groups_per_scale, n_latent_per_group=n_latent_per_group, scale_factor=scale_factor)
        self.groups = []
        for scale in range(n_latent_scales):
            n_groups = n_groups_per_scale[scale]
            for group in range(n_groups):
                output_channels = n_decoder_channels * mult
                if not(scale == 0 and group == 0):
                    group = Sequential()
                    for _ in range(res_cells_per_group):
                        group.add(GenerativeResidualCell(output_channels))
                    self.groups.append(group)
                self.groups.append(DecoderSampleCombiner(output_channels))
            
            if scale < n_latent_scales - 1:
                output_channels = n_decoder_channels * mult / scale_factor
                self.groups.append(
                    Rescaler(
                        output_channels, scale_factor=scale_factor, rescale_type=RescaleType.UP
                    )
                )
                mult /= scale_factor
        self.final_dec = GenerativeResidualCell(mult*n_decoder_channels)
        self.mult = mult

    def build(self, input_shape):
        self.batch_size = input_shape[0]

    def call(self, prior, enc_dec_combiners: List):
        z_params = []
        z0, params = self.sampler(prior, z_idx=0)
        z_params.append(params)
        h = tf.expand_dims(self.h, 0)
        h = tf.tile(h, [tf.shape(z0)[0], 1, 1, 1])
        x = self.groups[0](h, z0)
        
        combine_idx = 0
        for group in self.groups[1:]:
            if isinstance(group, DecoderSampleCombiner):
                prior = enc_dec_combiners[combine_idx](x)
                z_sample, params = self.sampler(prior, combine_idx+1)
                z_params.append(params)
                x = group(x, z_sample)
                combine_idx += 1
            else:
                x = group(x)
        
        return self.final_dec(x), z_params


class DecoderSampleCombiner(layers.Layer):
    def __init__(self, output_channels, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(output_channels, (1,1), strides=(1,1), padding="same")
    
    def call(self, x, z):
        output = tf.concat((x, z), axis=3)
        output = self.conv(output)
        return output


class GenerativeResidualCell(layers.Layer):
    """Generative network residual cell in NVAE architecture"""

    def __init__(self, output_channels, expansion_ratio=6, **kwargs):
        super().__init__(**kwargs)
        self.batch_norm1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(
            expansion_ratio * output_channels, (1, 1), padding="same"
        )
        self.batch_norm2 = layers.BatchNormalization()
        self.depth_conv = layers.DepthwiseConv2D((5, 5), padding="same")
        self.batch_norm3 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(output_channels, (1, 1), padding="same")
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
