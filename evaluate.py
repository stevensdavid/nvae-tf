import tensorflow as tf
import numpy as np
import skimage.transform
import scipy.linalg as scalg
import precision_recall as pr

# Takes 2 batches of images (b_size x 299 x 299 x 3) from different sources and calculates a FID score.
# Lower scores indicate closer resemblance in generated material to another data source.
# Inspired from: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
# TODO: support for batch size=1 (?) and progress logging
def fid_score(images1,images2):
    #images1,images2 = gen_images(70)
    act1,act2 = latent_activations(images1,images2)
	# model activations as gaussians 
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate distance
    dotp = sigma1.dot(sigma2)
    covmean = scalg.sqrtm(dotp)
    return np.sum((mu1-mu2)**2.0) + np.trace(sigma1+sigma2-2.0*(covmean.real))

# Evaluates the PR of images1 in reference to images2 using NVIDIAs implementation.
def precision_recall(images1,images2):
    act1,act2 = latent_activations(images1,images2)
    precision, recall = pr.knn_precision_recall_features(act1,act2)
    return precision, recall


# For comparing generated and real samples via Inception v3 latent representation.
# Returns latent activations from 2 sets of image batches.
def latent_activations(images1,images2):
    if not(images1.shape[1:] == (299,299,3) and images2.shape[1:] == (299,299,3)):
        images1,images2 = resize(images1), resize(images2)
    
    iv3_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet',pooling='avg',input_shape=(299,299,3))
    
	# latent representations
    act1 = iv3_model.predict(images1)
    act2 = iv3_model.predict(images2)
    return act1,act2

def gen_images(b_size):
    im1 = tf.random.normal(shape=[b_size,32,32,3],stddev=0.5,mean=0.0,dtype=tf.dtypes.float32)
    im2 = tf.random.normal(shape=[b_size,32,32,3],stddev=0.5,mean=0.0,dtype=tf.dtypes.float32)
    return im1,im2
    
def resize(images):
    print(images.shape)
    resized_images = []
    for img in images:
        print(img.shape)
        resized_images.append(skimage.transform.resize(img,(299,299,3),0))
    return tf.convert_to_tensor(resized_images, dtype=tf.float32)
