
from models import NVAE
from util import tile_images
import tensorflow as tf
import numpy as np
import skimage.transform
import scipy.linalg as scalg
import precision_recall as prec_rec
import perceptual_path_length as ppl
from tensorflow_probability import distributions


def save_samples_to_tensorboard(epoch, model, image_logger):
    for temperature in [0.7, 0.8, 0.9, 1.0]:
        images, *_ = model.sample(temperature=temperature)
        images = tile_images(images)
        images = tf.expand_dims(images, axis=0)
        with image_logger.as_default():
            tf.summary.image(
                f"t={temperature:.1f}", images, step=epoch,
            )


def save_reconstructions_to_tensorboard(epoch, model, test_data:tf.data.Dataset, image_logger):
    batch = tf.convert_to_tensor(next(test_data.shuffle(buffer_size=10).as_numpy_iterator())[0])
    # Tensorboard can only display 3 images
    batch = batch[:3]
    reconstruction_logits, *_ = model(batch)
    distribution = distributions.Bernoulli(
        logits=reconstruction_logits, dtype=tf.float32, allow_nan_stats=False
    )
    images = distribution.mean()
    comparison = tf.stack([batch, images], axis=1)
    b_sz, h, w, c = tf.shape(batch)
    # flip width and height
    comparison = tf.transpose(comparison, perm=[0, 1, 3, 2, 4])
    comparison = tf.reshape(comparison, [b_sz, 2 * w, h, c])
    # reset width and height order
    comparison = tf.transpose(comparison, perm=[0, 2, 1, 3])
    with image_logger.as_default():
        tf.summary.image("test_reconstruction", comparison, step=epoch)


def evaluate_model(epoch, model, test_data, metrics_logger, batch_size, n_attempts=1000):
    # PPL
    # slerp, slerp_perturbed = e.perceptual_path_length_init()
    # images1, images2 = model.sample(z=slerp), model.sample(z=slerp_perturbed)
    # TODO: Handle entire dataset
    # test_samples, _ = next(test_data.as_numpy_iterator())
    # test_samples = tf.convert_to_tensor(test_samples)
    with metrics_logger.as_default():
        # Negative log-likelihood
        nll = neg_log_likelihood(model, test_data, n_attempts=n_attempts)
        tf.summary.scalar("negative_log_likelihood", nll, step=epoch)
        for temperature in [0.7, 0.8, 0.9, 1.0]:
            # TODO: Handle batches, perform 1000 attempts and average
            temperature_scores = tf.convert_to_tensor([0., 0., 0., 0.])
            for attempt in range(n_attempts):
                attempt_scores = tf.convert_to_tensor([0., 0., 0., 0.])
                for test_batch, _ in test_data:
                    generated_images, last_s, z1, z2 = model.sample(
                        temperature=temperature, n_samples=batch_size
                    )
                    # PPL
                    slerp, slerp_perturbed = perceptual_path_length_init(z1, z2)
                    images1, images2 = (
                        model.sample_with_z(slerp, last_s),
                        model.sample_with_z(slerp_perturbed, last_s),
                    )
                    ppl = tf.reduce_mean(perceptual_path_length(images1, images2))
                    attempt_scores[0] += ppl
                    ppl = None
                    # PR
                    precision, recall = precision_recall(generated_images, test_batch)
                    attempt_scores[1] += precision
                    attempt_scores[2] += recall
                    # FID
                    fid = tf.reduce_mean(fid_score(generated_images, test_batch))
                    attempt_scores[3] += fid
                temperature_scores = attempt_scores / len(test_data) + temperature_scores
            temperature_scores = temperature_scores / n_attempts
            tf.summary.scalar(f"t={temperature}/ppl", temperature_scores[0], step=epoch)
            tf.summary.scalar(f"t={temperature}/precision", temperature_scores[1], step=epoch)
            tf.summary.scalar(f"t={temperature}/recall", temperature_scores[2], step=epoch)
            tf.summary.scalar(f"t={temperature}/fid", temperature_scores[3], step=epoch)
                


def neg_log_likelihood(model: NVAE, test_data: tf.data.Dataset, n_attempts=1000):
    nll = 0
    for batch, _ in test_data:
        batch_logs = []
        for _ in range(n_attempts):
            reconstruction, _, log_p, log_q = model(batch)
            log_iw = -model.calculate_recon_loss(batch, reconstruction) - log_q + log_p
            batch_logs.append(log_iw)
        nll -= tf.math.reduce_mean(
            tf.math.reduce_logsumexp(tf.stack(batch_logs), axis=0) - n_attempts
        )
    return nll / len(test_data)


# Takes 2 batches of images (b_size x 299 x 299 x 3) from different sources and calculates a FID score.
# Lower scores indicate closer resemblance in generated material to another data source.
# Inspired from: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
# TODO: support for batch size=1 (?) and progress logging
def fid_score(images1, images2):
    act1, act2 = latent_activations(images1, images2, "IV3")
    act1 = act1.numpy()
    act2 = act2.numpy()
    # model activations as gaussians
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate distance
    dotp = sigma1.dot(sigma2)
    covmean = scalg.sqrtm(dotp)
    return np.sum((mu1 - mu2) ** 2.0) + np.trace(sigma1 + sigma2 - 2.0 * (covmean.real))


# Evaluates the PR of images1 in reference to images2 using NVIDIAs implementation.
def precision_recall(images1, images2):
    act1, act2 = latent_activations(images1, images2, "IV3")
    # tf.compat.v1.disable_eager_execution()
    pr = prec_rec.knn_precision_recall_features(act1, act2)
    # tf.compat.v1.enable_eager_execution()
    return pr["precision"], pr["recall"]


# Calculates slerp from sampled latents. To continue PPL, generate images from the result of this function
# and call perceptual_path_length(images1,images2).
def perceptual_path_length_init(z1, z2, epsilon=1e-4):
    t = tf.random.uniform([tf.shape(z1)[0]], 0.0, 1.0)
    return ppl.slerp(z1, z2, t), ppl.slerp(z1, z2, t + epsilon)


# Takes generated images from interpolated latents and gives the PPL.
def perceptual_path_length(images1, images2):
    act1, _ = latent_activations(images1, images2, "VGG")

    return ppl.evaluate(act1)


# For comparing generated and real samples via Inception v3 latent representation.
# Returns latent activations from 2 sets of image batches.
def latent_activations(images1, images2, model_name):
    if not (images1.shape[1:] == (299, 299, 3) and images2.shape[1:] == (299, 299, 3)):
        images1, images2 = resize(images1), resize(images2)

    act1, act2 = 0, 0

    if model_name == "IV3":
        # TODO: Use
        model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=(299, 299, 3),
        )
        act1 = tf.convert_to_tensor(model.predict(images1,), dtype=tf.float32)
        act2 = tf.convert_to_tensor(model.predict(images2), dtype=tf.float32)
    elif model_name == "VGG":
        # TODO: Fix this! we don't have misc
        model = misc.load_pkl(
            "https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/vgg16_zhang_perceptual.pkl"
        )
        act1 = tf.convert_to_tensor(
            model.get_output_for(images1, images2), dtype=tf.float32
        )

    # latent representation

    return act1, act2


def gen_images(b_size, s1, s2, m1, m2):
    im1 = tf.random.normal(
        shape=[b_size, 32, 32, 3], stddev=s1, mean=m1, dtype=tf.dtypes.float32
    )
    im2 = tf.random.normal(
        shape=[b_size, 32, 32, 3], stddev=s2, mean=m2, dtype=tf.dtypes.float32
    )
    return im1, im2


def resize(images, target_shape=(299, 299, 3)):
    if tf.shape(images)[-1] == 1:
        images = tf.image.grayscale_to_rgb(images)
    resized_images = []
    for img in images:
        resized_images.append(skimage.transform.resize(img, target_shape, 0))
    return tf.convert_to_tensor(resized_images, dtype=tf.float32)


# -------For standalone debugging------
"""
def main():
    #a,b=gen_images(20,0.1,0.1,0,0)
    #print(a.shape)
    #p,r=precision_recall(a,b)
    #print(str(p) + " - " + str(r))

    a,b=gen_images(20,3,3,0,0)
    print(a.shape)
    p,r=precision_recall(a,b)
    print(str(p) + " - " + str(r))

if __name__ == "__main__":
    main()
#-------For standalone debugging------
"""

