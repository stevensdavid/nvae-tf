from argparse import ArgumentError, ArgumentParser
from evaluate import save_reconstructions_to_tensorboard
import os
import tensorflow as tf
from tensorflow.keras import callbacks
import random
import numpy as np


def main(args):
    print(f"Training with args: {args}")
    if args.cpu:
        tf.config.experimental.set_visible_devices([], "GPU")
    else:
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # Fix seeds
    tf.random.set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Imported here so seed can be set before imports
    from models import NVAE
    from evaluate import evaluate_model, save_samples_to_tensorboard

    if args.dataset == "mnist":
        from datasets import load_mnist

        train_data, test_data = load_mnist(batch_size=args.batch_size)
    else:
        raise ArgumentError("Unsupported dataset")
    if args.debug:
        train_data = train_data.take(4)  # DEBUG OPTION
        test_data = test_data.take(4)
    batches_per_epoch = len(train_data)
    model = NVAE(
        n_encoder_channels=args.n_encoder_channels,
        n_decoder_channels=args.n_decoder_channels,
        res_cells_per_group=args.res_cells_per_group,
        n_preprocess_blocks=args.n_preprocess_blocks,
        n_preprocess_cells=args.n_preprocess_cells,
        n_postprocess_blocks=args.n_postprocess_blocks,
        n_post_process_cells=args.n_postprocess_cells,
        n_latent_per_group=args.n_latent_per_group,
        n_latent_scales=len(args.n_groups_per_scale),
        n_groups_per_scale=args.n_groups_per_scale,
        sr_lambda=args.sr_lambda,
        scale_factor=args.scale_factor,
        total_epochs=args.epochs,
        n_total_iterations=len(train_data) * args.epochs,  # for balance kl
    )
    if args.resume_from > 0:
        model.load_weights(os.path.join(args.model_save_dir, f"{args.resume_from}"))
    lr_schedule = tf.keras.experimental.CosineDecay(
        initial_learning_rate=0.001, decay_steps=args.epochs * batches_per_epoch
    )
    adamax = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
    model.compile(optimizer=adamax, run_eagerly=True)

    image_logdir = os.path.join(args.tensorboard_log_dir, "images")
    image_logger = tf.summary.create_file_writer(image_logdir)
    metrics_logdir = os.path.join(args.tensorboard_log_dir, "metrics")
    metrics_logger = tf.summary.create_file_writer(metrics_logdir)

    def on_epoch_end(epoch, logs=None):
        if epoch % args.sample_frequency == 0:
            save_samples_to_tensorboard(epoch, model, image_logger)
            save_reconstructions_to_tensorboard(epoch, model, test_data, image_logger)
        # TODO: evaluate is buggy
        # if epoch % args.evaluate_frequency == 0:
        #     evaluate_model(epoch, model, test_data, metrics_logger, args.batch_size)

    training_callbacks = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(args.model_save_dir, "{epoch}"),
            save_freq=args.model_save_frequency * batches_per_epoch,
        ),
        callbacks.LambdaCallback(
            on_epoch_begin=model.on_epoch_begin,
            on_epoch_end=on_epoch_end,
        ),
    ]
    if args.patience:
        training_callbacks.append(
            callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True)
        )
    if args.tensorboard_log_dir:
        training_callbacks.append(
            callbacks.TensorBoard(
                log_dir=args.tensorboard_log_dir,
                update_freq=args.log_frequency * batches_per_epoch,
            )
        )

    model.fit(
        train_data,
        # validation_data=test_data,
        # validation_freq=args.log_frequency,
        epochs=args.epochs,
        callbacks=training_callbacks,
        initial_epoch=args.resume_from,
        verbose=1 if args.debug else 2,
    )
    # Statement crashes due to eager execution. Use final checkpoint instead.
    # model.save(os.path.join(args.model_save_dir, "final.tf"), save_format="tf")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=400, help="Number of epochs to train"
    )
    parser.add_argument("--batch_size", default=32, type=int)
    # Hyperparameters
    parser.add_argument(
        "--n_encoder_channels",
        type=int,
        default=32,
        help="Number of initial channels in encoder",
    )
    parser.add_argument(
        "--n_decoder_channels",
        type=int,
        default=32,
        help="Number of initial channels in decoder",
    )
    parser.add_argument(
        "--res_cells_per_group",
        type=int,
        default=1,
        help="Number of residual cells to use within each group",
    )
    parser.add_argument(
        "--n_preprocess_blocks",
        type=int,
        default=2,
        help="Number of blocks to use in the preprocessing layers",
    )
    parser.add_argument(
        "--n_preprocess_cells",
        type=int,
        default=3,
        help="Number of cells to use within each preprocessing block",
    )
    parser.add_argument(
        "--n_postprocess_blocks",
        type=int,
        default=2,
        help="Number of blocks to use in the postprocessing layers",
    )
    parser.add_argument(
        "--n_postprocess_cells",
        type=int,
        default=3,
        help="Number of cells to use within each postprocessing block",
    )
    parser.add_argument(
        "--n_latent_per_group",
        type=int,
        default=20,
        help="Number of latent stochastic variables to sample in each group",
    )
    parser.add_argument(
        "--n_groups_per_scale",
        nargs="+",
        default=[5, 10],
        help="Number of groups to include in each resolution scale",
    )
    parser.add_argument(
        "--sr_lambda", type=float, default=0.01, help="Spectral regularisation strength"
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
        "--debug", action="store_true", help="Use only first two batches of data"
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models",
        help="Directory to save models in",
    )
    parser.add_argument(
        "--resume_from", type=int, default=0, help="Epoch to resume training from"
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        type=str,
        default="logs",
        help="Directory to save Tensorboard logs in",
    )
    parser.add_argument(
        "--sample_frequency",
        type=int,
        default=10,
        help="Frequency in epochs to sample images which are stored in Tensorboard",
    )
    parser.add_argument(
        "--evaluate_frequency",
        type=int,
        default=10,
        help="Number of epochs between each model evaluation (FID, PPL etc)",
    )
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=1,
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
        default=10,
        help="Number of epochs between each model save",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed to use for initialization"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
