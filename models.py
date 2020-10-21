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
        self.v = []
        self.initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0)
        # Updated at start of each epoch
        self.epoch = 0
        self.total_epochs = total_epochs
        # Updated for each gradient pass, training step
        self.steps = 0

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
            recon_loss = self.calculate_recon_loss(data, reconstruction)
            spectral_loss, bn_loss = self.calculate_spectral_and_bn_loss()
            # warming up KL term for first 30% of training
            beta = min(self.epoch / 0.3 * self.total_epochs, 1)
            activate_balancing = (beta < 1)
            kl_loss = beta * self.calculate_kl_loss(z_params,activate_balancing)
            loss = tf.math.reduce_mean(recon_loss + kl_loss)
            total_loss = loss + spectral_loss + bn_loss
            # self.add_loss(loss+spectral_loss)
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.steps += 1
        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "spectral_loss": spectral_loss,
        }

    def calculate_kl_loss(self, z_params: List[DistributionParams], balancing):
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
            tf.convert_to_tensor(kl_per_group, dtype=tf.float32), axis=[0]
        )
		# balance kl
		if balancing:
            kl_alphas = calculate_kl_alphas(self.encoder.n_latent_scales,self.encoder.n_groups_per_scale)
			kl_coeffs = calculate_kl_coeff(0.0001)
            kl_all = tf.stack(kl_per_group, 1)
            kl_vals = tf.reduce_mean(kl_all,dim=0)
            kl_coeff_i = tf_reduce_mean(tf.math.abs(kl_all),dim=0) + 0.01
            total_kl = tf.reduce_sum(kl_coeff_i)
            kl_coeff_i = kl_coeff_i / kl_alphas * total_kl
            kl_coeff_i = kl_coeff_i / tf.reduce_mean(kl_coeff_i,1)
            kl = tf.reduce_sum(kl_all*kl_coeff_i,1)
            return kl
        else:
        	return loss
	
    # Calculates the balancer coefficients alphas. The coefficient decay for later groups,
    # for which original paper offer several functions. Here, a square function is used.
	def calculate_kl_alphas(self, num_scales, groups_per_scale):
        coeffs = []
        for i in range(num_scales):
			coeffs.append(np.square(2**i)/groups_per_scale[num_scales-i-1] * tf.ones([groups_per_scale[num_scales-i-1]],tf.float32))
		coeffs = tf.concat(coeffs, 0)
        coeffs /= tf.reduce_min(coeffs)
        return coeffs

	def calculate_kl_coeff(min_kl_coeff):
        #TODO: add as arg
        kl_const_portion = 0.0001
		warmup_iters = 0.3*self.n_total_iterations
        const_iters = kl_const_portion*self.n_total_iterations
        return max(min((self.steps - const_iters)/warmup_iters,1.0),min_kl_coeff)


    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def calculate_recon_loss(self, inputs, reconstruction):
        log_probs = distributions.Bernoulli(
            logits=reconstruction, dtype=tf.float32, allow_nan_stats=False
        ).log_prob(inputs)
        return -tf.math.reduce_sum(log_probs, axis=[1, 2, 3])

    def calculate_spectral_and_bn_loss(self):
        bn_loss = 0
        spectral_loss = 0
        spectral_index = 0

        def update_loss(layer):
            nonlocal spectral_loss, bn_loss, spectral_index
            if isinstance(layer, layers.Conv2D):
                w = layer.weights[0]
                w = tf.reshape(w, [tf.shape(w)[0], -1])
                if spectral_index == len(self.u):
                    d1, d2 = tf.shape(w)
                    self.u.append(
                        tf.Variable(self.initializer([1, d1]), trainable=False)
                    )
                    self.v.append(
                        tf.Variable(self.initializer([1, d2]), trainable=False)
                    )
                self.v[spectral_index] = tf.math.l2_normalize(
                    tf.squeeze(
                        tf.linalg.matmul(
                            tf.expand_dims(self.u[spectral_index], axis=1), w
                        ),
                        [1],
                    ),
                    axis=1,
                    epsilon=1e-3,
                )
                self.u[spectral_index] = tf.math.l2_normalize(
                    tf.squeeze(
                        tf.linalg.matmul(
                            w, tf.expand_dims(self.v[spectral_index], axis=2)
                        ),
                        [2],
                    ),
                    axis=1,
                    epsilon=1e-3,
                )
                sigma = tf.linalg.matmul(
                    tf.expand_dims(self.u[spectral_index], axis=1),
                    tf.linalg.matmul(w, tf.expand_dims(self.v[spectral_index], axis=2)),
                )
                spectral_loss += tf.math.reduce_sum(sigma)
                spectral_index += 1
            elif isinstance(layer, layers.BatchNormalization):
                bn_loss += tf.math.reduce_max(tf.math.abs(layer.weights[0]))
            elif hasattr(layer, "layers"):
                for inner_layer in layer.layers:
                    update_loss(inner_layer)

        for model in [self.encoder, self.decoder]:
            for layer in model.groups:
                update_loss(layer)
        return self.sr_lambda * spectral_loss, self.sr_lambda * bn_loss
