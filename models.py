import tensorflow as tf


class NVAE(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        return x
