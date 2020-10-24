import tensorflow as tf
import numpy as np
import skimage.transform
import scipy.linalg as scalg
import precision_recall as prec_rec
import perceptual_path_length as ppl

def save_samples_to_tensorboard(epoch, model, image_logger):
    for temperature in [0.7, 0.8, 0.9, 1.0]:
        images, *_ = model.sample(temperature=temperature)
        images = tile_images(images)
        images = tf.expand_dims(images, axis=0)
        with image_logger.as_default():
            tf.summary.image(
                f"t={temperature:.1f}",
                images,
                step=epoch,
            )


def evaluate_model(epoch, model, test_data, metrics_logger, batch_size):
    # PPL
    # slerp, slerp_perturbed = e.perceptual_path_length_init()
    # images1, images2 = model.sample(z=slerp), model.sample(z=slerp_perturbed)
    # TODO: Handle entire dataset
    test_samples, _ = next(test_data.as_numpy_iterator())
    with metrics_logger.as_default():
        # Negative log-likelihood
        nll = ...
        # tf.sumamry.scalar(f"negative_log_likelihood", nll, step=epoch)
        for temperature in [0.7, 0.8, 0.9, 1.0]:
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
            # PR
            precision, recall = precision_recall(generated_images, test_samples)
            # FID
            fid = tf.reduce_mean(fid_score(generated_images, test_samples))
            tf.sumamry.scalar(f"t={temperature}/ppl", ppl, step=epoch)
            tf.sumamry.scalar(f"t={temperature}/precision", precision, step=epoch)
            tf.sumamry.scalar(f"t={temperature}/recall", recall, step=epoch)
            tf.sumamry.scalar(f"t={temperature}/fid", fid, step=epoch)


# Takes 2 batches of images (b_size x 299 x 299 x 3) from different sources and calculates a FID score.
# Lower scores indicate closer resemblance in generated material to another data source.
# Inspired from: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
# TODO: support for batch size=1 (?) and progress logging
def fid_score(images1,images2):
    act1,act2 = latent_activations(images1,images2, 'IV3')
	# model activations as gaussians 
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate distance
    dotp = sigma1.dot(sigma2)
    covmean = scalg.sqrtm(dotp)
    return np.sum((mu1-mu2)**2.0) + np.trace(sigma1+sigma2-2.0*(covmean.real))

# Evaluates the PR of images1 in reference to images2 using NVIDIAs implementation.
def precision_recall(images1,images2):
    act1,act2 = latent_activations(images1,images2, 'IV3')
    #tf.compat.v1.disable_eager_execution()
    pr = prec_rec.knn_precision_recall_features(act1,act2)
    #tf.compat.v1.enable_eager_execution()
    return pr['precision'], pr['recall']

# Calculates slerp from sampled latents. To continue PPL, generate images from the result of this function
# and call perceptual_path_length(images1,images2).
def perceptual_path_length_init(z1, z2, epsilon=1e-4):
    t = tf.random_uniform([tf.shape(z1)[0]], 0.0, 1.0)
    return ppl.slerp(z1, z2, t), ppl.slerp(z1, z2, t + epsilon)

# Takes generated images from interpolated latents and gives the PPL.
def perceptual_path_length(images1, images2):
    act1, _ = latent_activations(images1, images2, 'VGG')
    
    return ppl.evaluate(act1)

# For comparing generated and real samples via Inception v3 latent representation.
# Returns latent activations from 2 sets of image batches.
def latent_activations(images1,images2, model_name):
    if not(images1.shape[1:] == (299,299,3) and images2.shape[1:] == (299,299,3)):
        images1,images2 = resize(images1), resize(images2)
    
    act1, act2 = 0,0

    if model_name == 'IV3':
    	model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet',pooling='avg',input_shape=(299,299,3))
        act1 = tf.convert_to_tensor(model.predict(images1, ),dtype=tf.float32)
    	act2 = tf.convert_to_tensor(model.predict(images2),dtype=tf.float32)
    elif model_name == 'VGG':
    	model = misc.load_pkl('https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/vgg16_zhang_perceptual.pkl')
        act1 = tf.convert_to_tensor(model.get_output_for(images1, images2),dtype=tf.float32)

	# latent representation
    
    return act1,act2

def gen_images(b_size,s1,s2,m1,m2):
    im1 = tf.random.normal(shape=[b_size,32,32,3],stddev=s1,mean=m1,dtype=tf.dtypes.float32)
    im2 = tf.random.normal(shape=[b_size,32,32,3],stddev=s2,mean=m2,dtype=tf.dtypes.float32)
    return im1,im2
    
def resize(images, target_shape=(299, 299, 3)):
    if tf.shape(images)[-1] == 1:
        images = tf.image.grayscale_to_rgb(images)
    print(images.shape)
    resized_images = []
    for img in images:
        print(img.shape)
        resized_images.append(skimage.transform.resize(img, target_shape, 0))
    return tf.convert_to_tensor(resized_images, dtype=tf.float32)

#-------For standalone debugging------
'''
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
'''