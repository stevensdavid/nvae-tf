from common import DistributionParams, Rescaler
from typing import List
from postprocess import Postprocess
from util import tile_images, softclamp5

# from tensorflow.python.training.tracking.data_structures import NonDependency
from decoder import Decoder, DecoderSampleCombiner
from encoder import Encoder
import tensorflow as tf
from tensorflow.keras import layers
from preprocess import Preprocess
from tensorflow_probability import distributions
import numpy as np


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
        total_epochs,
        n_total_iterations,
        step_based_warmup,
        input_shape,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sr_lambda = sr_lambda
        self.preprocess = Preprocess(
            n_encoder_channels,
            n_preprocess_blocks,
            n_preprocess_cells,
            scale_factor,
            input_shape,
        )
        self.n_latent_per_group = n_latent_per_group
        self.n_latent_scales = n_latent_scales
        self.n_groups_per_scale = n_groups_per_scale
        self.n_total_iterations = n_total_iterations
        self.n_preprocess_blocks = n_preprocess_blocks

        mult = self.preprocess.mult
        self.encoder = Encoder(
            n_encoder_channels=n_encoder_channels,
            n_latent_per_group=n_latent_per_group,
            res_cells_per_group=res_cells_per_group,
            n_latent_scales=n_latent_scales,
            n_groups_per_scale=n_groups_per_scale,
            mult=mult,
            scale_factor=scale_factor,
            input_shape=self.preprocess.output_shape_,
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
            input_shape=self.encoder.output_shape_,
        )
        mult = self.decoder.mult
        self.postprocess = Postprocess(
            n_postprocess_blocks,
            n_post_process_cells,
            scale_factor=scale_factor,
            mult=mult,
            n_channels_decoder=n_decoder_channels,
        )
        # Updated at start of each epoch
        self.epoch = 0
        self.total_epochs = total_epochs
        self.step_based_warmup = step_based_warmup
        # Updated for each gradient pass, training step
        self.steps = 0

    def call(self, inputs, nll=False):
        x = self.preprocess(inputs)
        enc_dec_combiners, final_x = self.encoder(x)
        # Flip bottom-up to top-down
        enc_dec_combiners.reverse()
        reconstruction, z_params, log_p, log_q = self.decoder(
            final_x, enc_dec_combiners, nll=nll
        )
        reconstruction = self.postprocess(reconstruction)
        return reconstruction, z_params, log_p, log_q

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
            reconstruction, z_params, *_ = self(data)
            recon_loss = self.calculate_recon_loss(data, reconstruction)
            bn_loss = self.calculate_bn_loss()
            # warming up KL term for first 30% of training
            warmup_metric = self.steps if self.step_based_warmup else self.epoch
            beta = min(warmup_metric / (0.3 * self.n_total_iterations), 1)
            activate_balancing = beta < 1
            kl_loss = beta * self.calculate_kl_loss(z_params, activate_balancing)
            loss = tf.math.reduce_mean(recon_loss + kl_loss)
            total_loss = loss + bn_loss
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.steps += 1
        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "bn_loss": bn_loss,
        }

    def sample(self, n_samples=16, temperature=1.0, greyscale=True):
        s = tf.expand_dims(self.decoder.h, 0)
        s = tf.tile(s, [n_samples, 1, 1, 1])
        z0_shape = tf.concat([[n_samples], self.decoder.z0_shape], axis=0)
        mu = softclamp5(tf.zeros(z0_shape))
        sigma = tf.math.exp(softclamp5(tf.zeros(z0_shape))) + 1e-2
        if temperature != 1.0:
            sigma *= temperature
        z = self.decoder.sampler.sample(mu, sigma)

        decoder_index = 0
        last_s = None
        # s should have shape 16,4,4,32
        # z should have shape 8,4,4,20
        for layer in self.decoder.groups:
            if isinstance(layer, DecoderSampleCombiner):
                if decoder_index > 0:
                    mu, log_sigma = self.decoder.sampler.get_params(
                        self.decoder.sampler.dec_sampler, decoder_index, s
                    )
                    mu = softclamp5(mu)
                    sigma = tf.math.exp(softclamp5(log_sigma)) + 1e-2
                    z = self.decoder.sampler.sample(mu, sigma)
                last_s = s
                s = layer(s, z)
                decoder_index += 1
            else:
                s = layer(s)

        reconstruction = self.postprocess(s)

        distribution = distributions.Bernoulli(
            logits=reconstruction, dtype=tf.float32, allow_nan_stats=False
        )
        if greyscale:
            images = distribution.probs_parameter()
        else:
            images = distribution.sample()
        z1 = self.decoder.sampler.sample(mu, sigma)
        z2 = self.decoder.sampler.sample(mu, sigma)
        # return images and mu, sigma, s used for sampling last hierarchical z in turn enabling sampling of images
        return images, last_s, z1, z2

    # As sample(), but starts from a fixed last hierarchical z given by mu, sigma and s. See sample() for details.
    def sample_with_z(self, z, s):
        last_gen_layer = self.decoder.groups[-1]
        s = last_gen_layer(s, z)
        reconstruction = self.postprocess(s)
        distribution = distributions.Bernoulli(
            logits=reconstruction, dtype=tf.float32, allow_nan_stats=False
        )
        images = distribution.mean()
        return images

    def calculate_kl_loss(self, z_params: List[DistributionParams], balancing):
        # -KL(q(z1|x)||p(z1)) - sum[ KL(q(zl|x,z<l) || p(z|z<l))]
        kl_per_group = []
        # n_groups x batch_size x 4
        loss = 0

        for g in z_params:
            term1 = (g.enc_mu - g.dec_mu) / g.dec_sigma
            term2 = g.enc_sigma / g.dec_sigma
            kl = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - tf.math.log(term2)
            kl_per_group.append(tf.math.reduce_sum(kl, axis=[1, 2, 3]))

        # balance kl
        if balancing:
            # Requires different treatment for encoder and decoder?
            kl_alphas = self.calculate_kl_alphas(
                self.n_latent_scales, self.n_groups_per_scale
            )
            kl_all = tf.stack(kl_per_group, 0)
            kl_coeff_i = tf.reduce_mean(tf.math.abs(kl_all), 1) + 0.01
            total_kl = tf.reduce_sum(kl_coeff_i)
            kl_coeff_i = kl_coeff_i / kl_alphas * total_kl
            kl_coeff_i = kl_coeff_i / tf.reduce_mean(kl_coeff_i, 0, keepdims=True)
            temp = tf.stack(kl_all, 1)
            # We stop gradient through kl_coeff_i because we are only interested
            # in changing the magnitude of the loss, not the direction of the
            # gradient.
            loss = tf.reduce_sum(temp * tf.stop_gradient(kl_coeff_i), axis=[1])
        else:
            loss = tf.math.reduce_sum(
                tf.convert_to_tensor(kl_per_group, dtype=tf.float32), axis=[0]
            )
        return loss

    # Calculates the balancer coefficients alphas. The coefficient decay for later groups,
    # for which original paper offer several functions. Here, a square function is used.
    def calculate_kl_alphas(self, num_scales, groups_per_scale):
        coeffs = []
        for i in range(num_scales):
            coeffs.append(
                np.square(2 ** i)
                / groups_per_scale[num_scales - i - 1]
                * tf.ones([groups_per_scale[num_scales - i - 1]], tf.float32,)
            )
        coeffs = tf.concat(coeffs, 0)
        coeffs /= tf.reduce_min(coeffs)
        return coeffs

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def calculate_recon_loss(self, inputs, reconstruction, crop_output=False):
        if crop_output:
            inputs = inputs[:, 2:30, 2:30, :]
            reconstruction = reconstruction[:, 2:30, 2:30, :]

        log_probs = distributions.Bernoulli(
            logits=reconstruction, dtype=tf.float32, allow_nan_stats=False
        ).log_prob(inputs)
        return -tf.math.reduce_sum(log_probs, axis=[1, 2, 3])

    def calculate_bn_loss(self):
        bn_loss = 0

        def update_loss(layer):
            nonlocal bn_loss
            if isinstance(layer, layers.BatchNormalization):
                bn_loss += tf.math.reduce_max(tf.math.abs(layer.weights[0]))
            elif hasattr(layer, "layers"):
                for inner_layer in layer.layers:
                    update_loss(inner_layer)

        for model in [self.encoder, self.decoder]:
            for layer in model.groups:
                update_loss(layer)

        return self.sr_lambda * bn_loss
