# import tensorflow as tf
# tf.config.experimental.set_visible_devices([], "GPU")
from argparse import ArgumentParser
import os
import tensorflow as tf
from tensorflow.keras import callbacks
import random
import numpy as np
from tensorflow.python.keras.engine import training


def main(args):
    tf.random.set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Imported here so seed can be set before imports
    from models import NVAE
    from datasets import load_mnist

    train_data, test_data = load_mnist(batch_size=8)

    if args.resume_from > 0:
        model = tf.keras.models.load_model(
            os.path.join(args.model_save_dir, f"{args.resume_from}.tf")
        )
    else:
        model = NVAE(
            n_encoder_channels=args.n_encoder_channels,
            n_decoder_channels=args.n_decoder_channels,
            res_cells_per_group=args.res_cells_per_group,
            n_preprocess_blocks=args.n_preprocess_blocks,
            n_preprocess_cells=args.n_preprocess_cells,
            n_postprocess_blocks=args.n_postprocess_blocks,
            n_post_process_cells=args.n_post_process_cells,
            n_latent_per_group=args.n_latent_per_group,
            n_latent_scales=len(args.n_groups_per_scale),
            n_groups_per_scale=args.n_groups_per_scale,
            sr_lambda=args.sr_lambda,
            scale_factor=args.scale_factor,
        )
        model.compile(optimizer="adamax", run_eagerly=True)
    training_callbacks = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(args.model_save_dir, "{epoch}.tf"),
            save_freq=args.model_save_frequency,
        )
    ]
    if args.patience:
        training_callbacks.append(
            callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True)
        )
    if args.tensorboard_log_dir:
        training_callbacks.append(
            callbacks.TensorBoard(
                log_dir=args.tensorboard_log_dir, update_freq=args.log_frequency
            )
        )

    model.fit(
        train_data,
        validation_data=test_data,
        validation_freq=args.log_frequency,
        epochs=args.epochs,
        callbacks=training_callbacks,
        initial_epoch=args.resume_from,
    )
    model.save(os.path.join(args.model_save_dir, "final.tf"), save_format="tf")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int)
    # Hyperparameters
    parser.add_argument(
        "--n_encoder_channels", type=int, help="Number of initial channels in encoder"
    )
    parser.add_argument(
        "--n_decoder_channels", type=int, help="Number of initial channels in decoder"
    )
    parser.add_argument(
        "--res_cells_per_group",
        type=int,
        help="Number of residual cells to use within each group",
    )
    parser.add_argument(
        "--n_preprocess_blocks",
        type=int,
        help="Number of blocks to use in the preprocessing layers",
    )
    parser.add_argument(
        "--n_preprocess_cells",
        type=int,
        help="Number of cells to use within each preprocessing block",
    )
    parser.add_argument(
        "--n_postprocess_blocks",
        type=int,
        help="Number of blocks to use in the postprocessing layers",
    )
    parser.add_argument(
        "--n_postprocess_cells",
        type=int,
        help="Number of cells to use within each postprocessing block",
    )
    parser.add_argument(
        "--n_latent_per_group",
        type=int,
        help="Number of latent stochastic variables to sample in each group",
    )
    parser.add_argument(
        "--n_groups_per_scale",
        nargs="+",
        default=[10, 5],
        help="Number of groups to include in each resolution scale",
    )
    parser.add_argument(
        "--sr_lambda", type=float, help="Spectral regularisation strength"
    )
    parser.add_argument(
        "--scale_factor",
        type=int,
        default=2,
        help="Factor to rescale image with in each scaling step",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist"],
        default="mnist",
        help="Dataset to use for training",
    )

    # Miscellaneous
    parser.add_argument("--cpu", action="store_true", help="Enforce CPU training")
    parser.add_argument(
        "--model_save_dir", type=str, help="Directory to save models in"
    )
    parser.add_argument(
        "--resume_from", type=int, default=0, help="Epoch to resume training from"
    )
    parser.add_argument(
        "--tensorboard_log_dir", type=str, help="Directory to save Tensorboard logs in"
    )
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=100,
        help="Number of epochs between each log write",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Early stopping patience threshold. Early stopping is only used if this is provided.",
    )
    parser.add_argument(
        "--model_save_frequency",
        type=int,
        default=10000,
        help="Number of epochs between each model save",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed to use for initialization"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
