from common import DistributionParams
from typing import List
from tensorflow.python.keras.backend import update
from tensorflow.python.keras.engine.sequential import Sequential
from postprocess import Postprocess
from tensorflow.python.keras.mixed_precision.experimental import loss_scale
from decoder import Decoder
from encoder import Encoder
import tensorflow as tf
from tensorflow.keras import layers
from preprocess import Preprocess
from tensorflow_probability import distributions
import os


def calculate_singular_value(w, iter=1):
    
    pass

def calculate_kl_loss(z_params: List[DistributionParams]):
    # z_params: enc_mu, enc_log_sigma, dec_mu, dec_log_sigma
    # -KL(q(z1|x)||p(z1)) - sum[ KL(q(zl|x,z<l) || p(z|z<l))]
    kl_per_group = []
    # n_groups x batch_size x 4
    loss = 0
    for g in z_params:
        enc_sigma = tf.math.exp(g.enc_log_sigma)
        dec_sigma = tf.math.exp(g.dec_log_sigma)

        term1 = (g.dec_mu - g.enc_mu) / enc_sigma
        term2 = dec_sigma / enc_sigma
        kl = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - tf.math.log(term2)
        kl_per_group.append(tf.math.reduce_sum(kl, axis=[1, 2, 3]))
    loss = tf.math.reduce_sum(
        tf.convert_to_tensor(
            kl_per_group,
            dtype=tf.float32,
            # dtype=tf.float16 if os.environ["MIXED_PRECISION"] else tf.float32,
        ),
        axis=[0],
    )

    return loss


def calculate_recon_loss(inputs, reconstruction):
    log_probs = distributions.Bernoulli(
        logits=reconstruction,
        allow_nan_stats=False,
    ).log_prob(inputs)
    return tf.cast(-tf.math.reduce_sum(log_probs, axis=[1, 2, 3]), tf.float32)


def calculate_spectral_and_bn_loss(lambda_, encoder, decoder, u):
    bn_loss = 0
    spectral_loss = 0
    spectral_index = 0
    def update_loss(layer,u):
        nonlocal spectral_loss, bn_loss, spectral_index
        if isinstance(layer, layers.Conv2D):
            w = layer.weights[0]
            v = tf.linalg.matmul(tf.transpose(w),u[spectral_index])
            v_hat = v / tf.math.l2_normalize(v)
            u_ = tf.linalg.matmul(w, v_hat)
            u_hat = u_ / tf.math.l2_normalize(u)
            sigma = tf.linalg.matmul(tf.linalg.matmul(tf.transpose(u_hat),w),v_hat)
            w_spec = w / tf.linalg.matmul(sigma, w)
            spectral_loss += tf.math.reduce_max(w_spec)
            u[spectral_index] = u_hat
            spectral_index += 1
        elif isinstance(layer, layers.BatchNormalization):
            bn_loss += tf.math.reduce_max(tf.math.abs(layer.weights[0]))
        elif hasattr(layer, "layers"):
            for inner_layer in layer.layers:
                update_loss(inner_layer)

    for model in [encoder, decoder]:
        for layer in model.groups:
            update_loss(layer,u)
    return lambda_ * spectral_loss, lambda_ * bn_loss


class NVAE(tf.keras.Model):
    def __init__(
        self,
        n_encoder_channels,
        n_decoder_channels,
        res_cells_per_group,
        n_preprocess_blocks,
        n_preprocess_cells,
        n_latent_per_group,
        n_latent_scales,
        n_groups_per_scale,
        n_postprocess_blocks,
        n_post_process_cells,
        sr_lambda,
        scale_factor,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sr_lambda = sr_lambda
        self.preprocess = Preprocess(
            n_encoder_channels, n_preprocess_blocks, n_preprocess_cells, scale_factor
        )
        mult = self.preprocess.mult
        self.encoder = Encoder(
            n_encoder_channels=n_encoder_channels,
            n_latent_per_group=n_latent_per_group,
            res_cells_per_group=res_cells_per_group,
            n_latent_scales=n_latent_scales,
            n_groups_per_scale=n_groups_per_scale,
            mult=mult,
            scale_factor=scale_factor,
        )
        mult = self.encoder.mult
        self.decoder = Decoder(
            n_decoder_channels=n_decoder_channels,
            n_latent_per_group=n_latent_per_group,
            res_cells_per_group=res_cells_per_group,
            n_latent_scales=n_latent_scales,
            n_groups_per_scale=list(reversed(n_groups_per_scale)),
            mult=mult,
            scale_factor=scale_factor,
        )
        mult = self.decoder.mult
        self.postprocess = Postprocess(
            n_postprocess_blocks,
            n_post_process_cells,
            scale_factor=scale_factor,
            mult=mult,
            n_channels_decoder=n_decoder_channels,
        )
        self.u = []
        for model in [self.encoder, self.decoder]:
            for layer in model.groups:
                if isinstance(layer, layers.Conv2D):
                    shape = tf.shape(layer.weights[0])
                    self.u.append(tf.Variable(tf.random_normal_initializer(shape), trainable=False))

    def call(self, inputs):
        x = self.preprocess(inputs)
        enc_dec_combiners, final_x = self.encoder(x)
        # Flip bottom-up to top-down
        enc_dec_combiners.reverse()
        reconstruction, z_params = self.decoder(final_x, enc_dec_combiners)
        reconstruction = self.postprocess(reconstruction)
        return reconstruction, z_params

    def train_step(self, data):
        """Training step for NVAE

        Args:
            data (Union[tf.Tensor, Tuple[tf.Tensor, Any]]): Labeled or unlabeled images

        Returns:
            dict[str, float]: All loss values

        Notes
        =====
        Adapted from Keras tutorial https://keras.io/examples/generative/vae/
        """
        if isinstance(data, tuple):
            # We have labeled data. Remove the label.
            data = data[0]
        with tf.GradientTape() as tape:
            reconstruction, z_params = self(data)
            kl_loss = calculate_kl_loss(z_params)
            recon_loss = calculate_recon_loss(data, reconstruction)
            spectral_loss, bn_loss = calculate_spectral_and_bn_loss(
                self.sr_lambda, self.encoder, self.decoder, self.u
            )
            loss = tf.math.reduce_mean(recon_loss + kl_loss)
            total_loss = loss + spectral_loss + bn_loss
            # self.add_loss(loss+spectral_loss)
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "spectral_loss": spectral_loss,
        }

    def sample(self, n_samples, t):
        pass
