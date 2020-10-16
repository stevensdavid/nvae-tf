from decoder import Decoder
from encoder import Encoder
import tensorflow as tf
from tensorflow.keras import layers
from preprocess import Preprocess


SCALE_FACTOR = 2


class NVAE(tf.keras.Model):
    def __init__(
        self,
        n_encoder_channels,
        res_cells_per_group,
        n_groups,
        n_preprocess_blocks,
        n_preprocess_cells,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.preprocess = Preprocess(
            n_encoder_channels, n_preprocess_blocks, n_preprocess_cells
        )
        mult = self.preprocess.mult
        self.encoder = Encoder(
            n_encoder_channels=3,
            n_latent_per_group=1,
            res_cells_per_group=2,
            n_latent_scales=2,
            n_groups_per_scale=[3, 1],
            mult=mult,
        )
        mult = self.encoder.mult
        self.decoder = Decoder()

    def call(self, x):
        x = self.preprocess(x)
        group_outputs, combiners, final_x = self.encoder(x)
        dist_params = self.encoder.sampler[0](final_x)
        mu, log_sigma = tf.split(dist_params, 2, axis=-1)
        mu = tf.squeeze(mu)
        log_sigma = tf.squeeze(log_sigma)
        # z = tf.random.normal(shape=mu.shape, mean=mu, stddev=tf.math.exp(log_sigma))
        # reparametrization trick
        z = mu + tf.random.normal(shape=mu.shape) * tf.math.exp(log_sigma)
        return x

    def sample(self):
        pass

