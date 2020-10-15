import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
from models import NVAE
from datasets import load_mnist

if __name__ == "__main__":
    train_data, test_data = load_mnist(batch_size=8)
    model = NVAE(n_encoder_channels=32, res_cells_per_group=2, n_groups=15)
    for batch_x, _ in train_data:
        res = model(batch_x)
    pass
