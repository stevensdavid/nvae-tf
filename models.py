from decoder import Decoder
from encoder import Encoder
import tensorflow as tf
from tensorflow.keras import layers
from preprocess import Preprocess


class NVAE(tf.keras.Model):
    def __init__(
        self,
        n_encoder_channels,
        n_decoder_channels,
        res_cells_per_group,
        n_groups,
        n_preprocess_blocks,
        n_preprocess_cells,
        n_latent_per_group,
        n_latent_scales,
        n_groups_per_scale,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.preprocess = Preprocess(
            n_encoder_channels, n_preprocess_blocks, n_preprocess_cells
        )
        mult = self.preprocess.mult
        self.encoder = Encoder(
            n_encoder_channels=n_encoder_channels,
            n_latent_per_group=n_latent_per_group,
            res_cells_per_group=res_cells_per_group,
            n_latent_scales=n_latent_scales,
            n_groups_per_scale=n_groups_per_scale,
            mult=mult,
        )
        # self.sampler = Sampler(n_latent_scales=n_latent_scales, n_groups_per_scale=n_groups_per_scale, n_latent_per_group=n_latent_per_group)
        mult = self.encoder.mult
        self.decoder = Decoder(
            n_decoder_channels=n_decoder_channels,
            n_latent_per_group=n_latent_per_group,
            res_cells_per_group=res_cells_per_group,
            n_latent_scales=n_latent_scales,
            n_groups_per_scale=list(reversed(n_groups_per_scale)),
            mult=mult,
            # sampler=self.sampler
        )

    def call(self, input):
        x = self.preprocess(input)
        enc_dec_combiners, final_x = self.encoder(x)
        # Flip bottom-up to top-down
        enc_dec_combiners.reverse()
        # z0 = self.decoder.sampler(prior=final_x, z_idx=0)
        # reconstruction = self.decoder(z0, enc_dec_combiners)
        reconstruction, z_params = self.decoder(final_x, enc_dec_combiners)
        return reconstruction, z_params

    def sample(self, n_samples, t):
        pass

