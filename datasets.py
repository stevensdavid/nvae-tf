import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

def load_mnist(batch_size):
    train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], batch_size=batch_size, as_supervised=True)
    def transform(image, label):
        image = tf.image.resize_with_crop_or_pad(image, 32, 32)
        image = tf.cast(image, dtype=tf.float32)
        image = tfp.distributions.Bernoulli(probs=image).sample()
        return image, label

    return train_ds.map(transform), test_ds.map(transform)

def load_celeba():
    # tfds.load('celeb_a')
    pass

if __name__ == "__main__":
    load_celeba()
