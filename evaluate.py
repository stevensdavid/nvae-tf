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
#from pympler import muppy, summary
from tqdm import tqdm
import importlib

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
        precision, recall = 0, 0
        overall_ppl = 0
        initial_attempts = 0 #before OOM crashing
        
        rescaled_test_data = test_data.map(lambda x, _: tf.py_function(resize, [x], Tout=tf.float32))
        for attempt in range(0, n_attempts):
            generated_images, last_s, z1, z2, _ = model.sample(
                temperature=temperature, n_samples=batch_size
            )

            for i, test_batch in enumerate(rescaled_test_data):
                print("BATCH %d out of %d | ATTEMPT %d out of %d" % (i, len(test_data), attempt + initial_attempts, n_attempts))

                # PR
                
                pr_images, *_ = model.sample(temperature=temperature, n_samples=tf.shape(test_batch)[0])
                test_batch, *_ = model.sample(temperature=temperature, n_samples=tf.shape(test_batch)[0])
                batch_precision, batch_recall = precision_recall(
                    pr_images, test_batch
                )
                """  
                # PPL
                slerp, slerp_perturbed = perceptual_path_length_init(z1, z2)
                images1, images2 = (
                    model.sample_with_z(slerp, last_s),
                    model.sample_with_z(slerp_perturbed, last_s),
                )
                batch_ppl = tf.reduce_mean(perceptual_path_length(images1, images2))
                """
                batch_ppl = 0
                
                """
                # Save progress
                resfile = open("res.txt", "r")
                lines = resfile.readlines()
                # Num of iterations before stopping. 
                
                if lines == []:
                    p, r, ppl_prev, prev_iterations = 0,0,0,0
                else:
                    p, r, ppl_prev, prev_iterations = lines
                    p = float(p)
                    r = float(r)
                    ppl_prev = float(ppl_prev)
                    prev_iterations = int(prev_iterations)

                print("PR %f %f  PPL %f" % (batch_precision, batch_recall, batch_ppl))

                p = (p + batch_precision)
                r = (r + batch_recall)
                ppl_new = (batch_ppl + ppl_prev)
                ppl_new = 0
                performed_iterations = prev_iterations + 1
                resfile.close()
                """
                
                tot_iterations = n_attempts*len(rescaled_test_data)
                curr_iterations = attempt*len(rescaled_test_data) + i
                
                print("Stopping condition: %d out of %d" % (curr_iterations, tot_iterations))
                """
                resfile = open("res.txt", "w")
                resfile.write(f"{p}\n")
                resfile.write(f"{r}\n")
                resfile.write(f"{ppl_new}\n")
                resfile.write(f"{performed_iterations}\n")
                resfile.close()
                """

                #if performed_iterations >= tot_iterations:
                """
                ppl_new = ppl_new / performed_iterations
                p = p / performed_iterations
                r = r / performed_iterations
                precision, recall, overall_ppl = p, r, ppl_new
                """
                    

                record = open("record.txt", "a")
                record.write(f"{batch_precision}\n")
                record.write(f"{batch_recall}\n")
                record.close()

        record = open("record.txt", "r")
        lines = record.readlines()
        record_precision = [float(lines[i]) for i in range(len(lines)) if i%2 == 0]
        record_recall = [float(lines[i]) for i in range(len(lines)) if i%2 == 1]

        evaluation.sample_metrics.append(
            Metrics(
                temperature=temperature,
                fid=None,
                ppl=None,
                precision=Metric.from_list(record_precision),
                recall=Metric.from_list(record_recall)
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
    return pr["precision"][0], pr["recall"][0]


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

