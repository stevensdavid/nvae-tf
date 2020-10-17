from postprocess import Postprocess
from tensorflow.python.keras.mixed_precision.experimental import loss_scale
from decoder import Decoder
from encoder import Encoder
import tensorflow as tf
from tensorflow.keras import layers
from preprocess import Preprocess
from tensorflow_probability import distributions


def calculate_kl_loss(z_params):
    # z_params: enc_mu, enc_log_sigma, dec_mu, dec_log_sigma
    # -KL(q(z1|x)||p(z1)) - sum[ KL(q(zl|x,z<l) || p(z|z<l))]
    kl_per_group = []
    #---------TODO: Calculate first term of KL--------------
    #n_groups x batch_size x 4
    for g in z_params[1:]:
        enc_sigma = tf.math.exp(g.enc_log_sigma)
        dec_sigma = tf.math.exp(g.dec_log_sigma)
        # batch_size x H x W x C
        kl = 0.5*(g.dec_mu**2/enc_sigma**2 + dec_sigma**2 - g.dec_log_sigma**2 - 1)
        kl = tf.math.reduce_sum(kl, axis=[1,2,3])
        kl_per_group.append(kl)
    loss = tf.math.reduce_sum(tf.convert_to_tensor(kl_per_group, dtype=tf.float32), axis=[0])
    
    return loss

def calculate_recon_loss(input, reconstruction):
    log_probs = distributions.Bernoulli(
        logits=reconstruction,
        dtype=tf.float32,
        allow_nan_stats=False
    ).log_prob(input)
    return -tf.math.reduce_sum(log_probs, axis=[1,2,3])

def calculate_spectral_loss(lambda_, encoder, decoder):
    loss = 0
    for layer in encoder.groups:
        if isinstance(layer, layers.Conv2D):
            weights = layer.weights
            loss += tf.math.reduce_max(weights)
    return lambda_ * loss

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
        n_postprocess_blocks,
        n_post_process_cells,
        lambda_,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lambda_ = lambda_
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
        mult = self.decoder.mult
        self.postprocess = Postprocess(n_postprocess_blocks, n_post_process_cells, mult, n_decoder_channels)

    def call(self, input):
        x = self.preprocess(input)
        enc_dec_combiners, final_x = self.encoder(x)
        # Flip bottom-up to top-down
        enc_dec_combiners.reverse()
        # z0 = self.decoder.sampler(prior=final_x, z_idx=0)
        # reconstruction = self.decoder(z0, enc_dec_combiners)
        reconstruction, z_params = self.decoder(final_x, enc_dec_combiners)
        reconstruction = self.postprocess(reconstruction)
        kl_loss = calculate_kl_loss(z_params)
        recon_loss = calculate_recon_loss(input, reconstruction)
        spectral_loss = calculate_spectral_loss(self.lambda_, self.encoder, self.decoder)
        loss = tf.math.reduce_mean(recon_loss+kl_loss)
        self.add_loss(loss+spectral_loss)
        return reconstruction

    def sample(self, n_samples, t):
        pass

