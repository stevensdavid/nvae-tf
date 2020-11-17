from numpy.lib.arraysetops import isin
from fid_utils import calculate_fid_given_paths
from models import NVAE
from util import Metric, Metrics, ModelEvaluation, sample_to_dir, save_images_to_dir, tile_images
import tensorflow as tf
import numpy as np
import skimage.transform
import scipy.linalg as scalg
import precision_recall as prec_rec
import perceptual_path_length as ppl
from tensorflow_probability import distributions
import os
from tqdm import tqdm, trange

model_iv3, model_vgg = None, None

def save_samples_to_tensorboard(epoch, model, image_logger):
    for temperature in [0.7, 0.8, 0.9, 1.0]:
        images, *_ = model.sample(temperature=temperature, n_samples=3)
        with image_logger.as_default():
            tf.summary.image(
                f"t={temperature:.1f}", images, step=epoch,
            )


def save_reconstructions_to_tensorboard(
    epoch, model, test_data: tf.data.Dataset, image_logger
):
    batch = tf.convert_to_tensor(
        next(test_data.shuffle(buffer_size=10).as_numpy_iterator())[0]
    )
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


def evaluate_model(
    epoch, model, test_data, metrics_logger, batch_size, n_attempts=10
) -> ModelEvaluation:
    test_data = test_data.shuffle(batch_size)
    evaluation = ModelEvaluation(nll=None, sample_metrics=[])
    ppl_attempts = 1000
    for temperature in tqdm(
        [1.0], desc="Temperature based tests (PPL/PR)", total=4
    ):
        ppls = []
        precisions = []
        recalls = []
        for attempt in trange(n_attempts, desc="PR Attempts"):
            batch_precisions = []
            batch_recalls = []
            for test_batch, _ in tqdm(test_data, desc="Batch", total=len(test_data)):
                # PR
                pr_images, *_ = model.sample(temperature=temperature, n_samples=tf.shape(test_batch)[0])
                batch_precision, batch_recall = precision_recall(
                    pr_images, test_batch
                )
                batch_precisions.append(batch_precision)
                batch_recalls.append(batch_recall)
            
            precision = np.mean(batch_precisions)
            recall = np.mean(batch_recalls)
            precisions.append(precision)
            recalls.append(recall)
        for attempt in trange(ppl_attempts, desc="PPL"):
            z1 = model.sample_z0(n_samples=batch_size, temperature=temperature)
            z2 = model.sample_z0(n_samples=batch_size, temperature=temperature)
            # generated_images, last_s, z1, z2 = model.sample(
            #     temperature=temperature, n_samples=batch_size
            # )
            # PPL
            slerp, slerp_perturbed = perceptual_path_length_init(z1, z2)
            # images1, images2 = (
            #     model.sample_with_z(slerp, last_s),
            #     model.sample_with_z(slerp_perturbed, last_s),
            # )
            images1, images2 = (
                model.sample(n_samples=batch_size, temperature=temperature, z=slerp),
                model.sample(n_samples=batch_size, temperature=temperature, z=slerp_perturbed),
            )
            batch_ppl = tf.reduce_mean(perceptual_path_length(images1, images2))
            ppl = np.mean(batch_ppl)
            ppls.append(ppl)

        evaluation.sample_metrics.append(
            Metrics(
                temperature=temperature,
                fid=None,
                ppl=Metric.from_list(ppls),
                precision=None,#Metric.from_list(precisions),
                recall=None,#Metric.from_list(recalls)
            )
        )
    # Negative log-likelihood
    # evaluation.nll = neg_log_likelihood(model, test_data, n_attempts=n_attempts)
    return evaluation


def neg_log_likelihood(model: NVAE, test_data: tf.data.Dataset, n_attempts=10):
    nlls = []
    for batch, _ in tqdm(test_data, desc="NLL Batch", total=len(test_data)):
        batch_logs = []
        for _ in range(n_attempts):
            reconstruction, _, log_p, log_q = model(batch, nll=True)
            log_iw = -model.calculate_recon_loss(batch, reconstruction, crop_output=True) - log_q + log_p
            batch_logs.append(log_iw)
        nll = -tf.math.reduce_mean(
            tf.math.reduce_logsumexp(tf.stack(batch_logs), axis=0) - tf.math.log(float(n_attempts))
        )
        nlls.append(nll)
    return Metric.from_list(nlls)


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


def evaluate_fid(model: NVAE, dataset, dataset_name, batch_size, temperature):
    dataset_dir = os.path.join("images", dataset_name, "actual")
    output_dir = os.path.join("images", dataset_name, f"generated_t_{temperature}")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    if not os.listdir(dataset_dir):
        # We need to save the source images to the directory
        for image_batch, _ in tqdm(dataset, desc="Saving dataset (FID)"):
            save_images_to_dir(image_batch, dataset_dir)
    for filename in os.listdir(output_dir):
        # Delete all old generated images
        os.remove(os.path.join(output_dir, filename))
    # Recommended by FID author
    sample_size = 10000
    sample_to_dir(model, batch_size, sample_size, temperature, output_dir)
    os.makedirs("fid", exist_ok=True)
    print("[FID] Calculating FID")
    fid_value = calculate_fid_given_paths(
        [dataset_dir, output_dir], inception_path="fid"
    )
    return fid_value


# Evaluates the PR of images1 in reference to images2 using NVIDIAs implementation.
def precision_recall(images1, images2):
    act1, act2 = latent_activations(images1, images2, "VGG")
    # tf.compat.v1.disable_eager_execution()
    # act1 = tf.reshape(act1, (tf.shape(act1)[0], -1))
    # act2 = tf.reshape(act1, (tf.shape(act2)[0], -1))
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
    act1, act2 = latent_activations(images1, images2, "VGG")
    return ppl.evaluate(act1, act2)


# For comparing generated and real samples via Inception v3 latent representation.
# Returns latent activations from 2 sets of image batches.
def latent_activations(images1, images2, model_name):
    global model_iv3, model_vgg
    if not (images1.shape[1:] == (299, 299, 3) and images2.shape[1:] == (299, 299, 3)):
        images1, images2 = resize(images1), resize(images2)

    act1, act2 = 0, 0
    if model_name == "IV3":
        # TODO: Use
        if model_iv3 is None:
            model_iv3 = tf.keras.applications.InceptionV3(
                include_top=False,
                weights="imagenet",
                pooling="avg",
                input_shape=(299, 299, 3),
            )
        act1 = tf.convert_to_tensor(model_iv3.predict(images1,), dtype=tf.float32)
        act2 = tf.convert_to_tensor(model_iv3.predict(images2), dtype=tf.float32)
    elif model_name == "VGG":
        if model_vgg is None:
            model_vgg = tf.keras.applications.VGG16(include_top=False, pooling="avg")

        # print("------------ FIRST ------------")
        # all_objects = muppy.get_objects()
        # sum1 = summary.summarize(all_objects) 
        # summary.print_(sum1)
        
        act1 = model_vgg(images1)
        act2 = model_vgg(images2) # OOM

    """
    print("------------ SECOND ------------")
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    summary.print_(sum1)
    """

    """
    print("------------ THIRD ------------")
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    summary.print_(sum1)
    """
    
    """
    for d in all_objects:
        if(isinstance(d, pd.DataFrame)):
            print(d.columns.values)
            print(len(d))
    """
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
        if isinstance(img, tf.Tensor):
            img = img.numpy()
        resized_images.append(skimage.transform.resize(img, target_shape, 0))
    return tf.convert_to_tensor(resized_images, dtype=tf.float32)


# -------For standalone debugging------


def main():
    # a,b=gen_images(20,0.1,0.1,0,0)
    # print(a.shape)
    # p,r=precision_recall(a,b)
    # print(str(p) + " - " + str(r))

    a, b = gen_images(20, 3, 3, 0, 0)
    print(a.shape)
    p, r = precision_recall(a, b)
    print(str(p) + " - " + str(r))


if __name__ == "__main__":
    main()
# -------For standalone debugging------

