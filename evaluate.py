import tensorflow as tf
import numpy as np
import skimage.transform
import scipy.linalg as scalg
import precision_recall as prec_rec

# Takes 2 batches of images (b_size x 299 x 299 x 3) from different sources and calculates a FID score.
# Lower scores indicate closer resemblance in generated material to another data source.
# Inspired from: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
# TODO: support for batch size=1 (?) and progress logging
def fid_score(images1,images2):
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
    #tf.compat.v1.disable_eager_execution()
    pr = prec_rec.knn_precision_recall_features(act1,act2)
    #tf.compat.v1.enable_eager_execution()
    return pr['precision'], pr['recall']

# For comparing generated and real samples via Inception v3 latent representation.
# Returns latent activations from 2 sets of image batches.
def latent_activations(images1,images2):
    if not(images1.shape[1:] == (299,299,3) and images2.shape[1:] == (299,299,3)):
        images1,images2 = resize(images1), resize(images2)
    
    iv3_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet',pooling='avg',input_shape=(299,299,3))
    
	# latent representations
    act1 = tf.convert_to_tensor(iv3_model.predict(images1),dtype=tf.float32)
    act2 = tf.convert_to_tensor(iv3_model.predict(images2),dtype=tf.float32)
    return act1,act2

def gen_images(b_size,s1,s2,m1,m2):
    im1 = tf.random.normal(shape=[b_size,32,32,3],stddev=s1,mean=m1,dtype=tf.dtypes.float32)
    im2 = tf.random.normal(shape=[b_size,32,32,3],stddev=s2,mean=m2,dtype=tf.dtypes.float32)
    return im1,im2
    
def resize(images):
    print(images.shape)
    resized_images = []
    for img in images:
        print(img.shape)
        resized_images.append(skimage.transform.resize(img,(299,299,3),0))
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