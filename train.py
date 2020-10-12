import tensorflow as tf
from datasets import load_mnist
from tensorflow.keras import optimizers, experimental

def train_mnist():
    _lambda = 1e-2
    init_lr = 0.01
    epochs = 400
    batch_size = 200
    steps = int(60000/batch_size)
    latent_variable_scales = [(5,4),(10,8)]
    n_channels = 20
    init_channels = 32
    train_ds, test_ds = load_mnist(batch_size)
    # cosine_decay = optimizers.schedules.LearningRateSchedule(experimental.CosineDecay(init_lr, steps, alpha=0.0,name='CosineDecay'))
    # optimizer = optimizers.Adamax(learning_rate=cosine_decay, name='Adamax')

    for epoch in range(epochs):
        for (xs,ys) in train_ds:
            pass
    
if __name__ == "__main__":
    train_mnist()