import tensorflow as tf
import tensorflow_datasets as tfds


def load_mnist(batch_size):
    train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], batch_size=batch_size, as_supervised=True)
    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255, label

    return train_ds.map(normalize), test_ds.map(normalize)

def load_celeba():
    # tfds.load('celeb_a')
    pass

if __name__ == "__main__":
    load_celeba()
