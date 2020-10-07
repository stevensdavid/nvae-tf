import tensorflow as tf
import tensorflow_datasets as tfds


def load_mnist(batch_size):
    train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], batch_size=batch_size, as_supervised=True)
    return train_ds, test_ds

def load_celeba():
    # tfds.load('celeb_a')

if __name__ == "__main__":
    load_celeba()
