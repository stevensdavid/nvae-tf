from argparse import ArgumentError, ArgumentParser
from random import choice
from util import sample_to_dir, save_image_to_dir
from evaluate import save_reconstructions_to_tensorboard
import os
import tensorflow as tf
from tensorflow.keras import callbacks
import random
import numpy as np
import pickle


def checkpoint_path(model_save_dir, epoch):
    return os.path.join(model_save_dir, f"epoch_{epoch}")


def train(args, model, train_data, test_data):
    from evaluate import save_samples_to_tensorboard

    image_logdir = os.path.join(args.tensorboard_log_dir, "images")
    image_logger = tf.summary.create_file_writer(image_logdir)

    def on_epoch_end(epoch, logs=None):
        if epoch % args.sample_frequency == 0:
            save_samples_to_tensorboard(epoch, model, image_logger)
            save_reconstructions_to_tensorboard(epoch, model, test_data, image_logger)
        if epoch % args.model_save_frequency == 0:
            model.save_weights(checkpoint_path(args.model_save_dir, epoch))

    training_callbacks = [
        callbacks.LambdaCallback(
            on_epoch_begin=model.on_epoch_begin, on_epoch_end=on_epoch_end,
        ),
    ]
    if args.patience:
        training_callbacks.append(
            callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True)
        )
    if args.tensorboard_log_dir:
        training_callbacks.append(
            callbacks.TensorBoard(
                log_dir=args.tensorboard_log_dir, update_freq="epoch",
            )
        )

    model.fit(
        train_data,
        epochs=args.epochs,
        callbacks=training_callbacks,
        initial_epoch=args.resume_from,
        verbose=1 if args.debug or args.verbose else 2,
        workers=args.workers,
        use_multiprocessing=args.multiprocessing,
    )
    model.save_weights(checkpoint_path(args.model_save_dir, "final"))


def test(args, model, test_data):
    from evaluate import evaluate_model

    metrics_logdir = os.path.join(args.tensorboard_log_dir, "metrics")
    metrics_logger = tf.summary.create_file_writer(metrics_logdir)
    print("Calling evaluate...")
    evaluation = evaluate_model(
        epoch=args.resume_from,
        model=model,
        test_data=test_data,
        metrics_logger=metrics_logger,
        batch_size=args.batch_size,
        n_attempts=10,
        binary=args.binary_eval,
    )
    print(f"Negative log likelihood: {evaluation.nll}")
    print(evaluation)


def sample(args, model):    
    for t in [0.7, 0.8, 0.9, 1]:
        output_dir = os.path.join(args.sample_dir, f"t_{t:.1f}")
        os.makedirs(output_dir, exist_ok=True)
        sample_to_dir(model, args.batch_size, args.n_samples, t, output_dir)


def interpolate(args, model):
    for _ in range(args.n_interpolations):
        interpolation = model.interpolate(n_steps=args.interpolation_steps)
        n, h, w, c = tf.shape(interpolation)
        interpolation = tf.transpose(interpolation, perm=[0,2,1,3])
        interpolation = tf.reshape(interpolation, [n*w, h, c])
        interpolation = tf.transpose(interpolation, [1,0,2])
        save_image_to_dir(interpolation, args.interpolation_dir)


def main(args):
    print(f"Args: {args}")
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

    if args.dataset == "mnist":
        from datasets import load_mnist

        train_data, test_data = load_mnist(batch_size=args.batch_size, binary=args.mode == "train" or args.binary_eval)
    else:
        raise ArgumentError("Unsupported dataset")
    if args.debug:
        train_data = train_data.take(4)  # DEBUG OPTION
        test_data = test_data.take(4)
    batches_per_epoch = len(train_data)

    sample_batch, sample_labels = next(train_data.as_numpy_iterator())

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
        step_based_warmup=args.step_based_warmup,
        input_shape=tf.convert_to_tensor(sample_batch.shape, dtype=float),
    )
    lr_schedule = tf.keras.experimental.CosineDecay(
        initial_learning_rate=0.001, decay_steps=args.epochs * batches_per_epoch
    )
    adamax = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
    model.compile(optimizer=adamax, run_eagerly=True)
    if args.resume_from > 0:
        model.load_weights(checkpoint_path(args.model_save_dir, args.resume_from))
        model.steps = args.resume_from * args.batch_size

    if args.mode == "train":
        train(args, model, train_data, test_data)
    elif args.mode == "test":
        test(args, model, test_data)
    elif args.mode == "sample":
        sample(args, model)
    elif args.mode == "interpolate":
        interpolate(args, model)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=400, help="Number of epochs to train"
    )
    parser.add_argument("--batch_size", default=144, type=int)
    parser.add_argument("--mode", type=str, choices=["train", "test", "sample", "interpolate"])
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
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples to generate in sample mode",
    )
    parser.add_argument(
        "--n_interpolations", type=int, default=10, help="Number of interpolations to perform."
    )
    parser.add_argument(
        "--interpolation_steps", type=int, default=10, help="Number of steps to interpolate."
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models",
        help="Directory to save models in",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default="results",
        help="Directory to save sampled images in. Only applicable in sample mode.",
    )
    parser.add_argument(
        "--interpolation_dir",
        type=str,
        default="interp",
        help="Directory to save interpolated images in. Only applicable in interpolation mode."
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
        default=5,
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
    parser.add_argument("--binary_eval", action="store_true", help="Evaluate on binary data")
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
        "--step_based_warmup",
        action="store_true",
        help="Base warmup on batches trained instead of epochs",
    )
    parser.add_argument("--workers", default=1)
    parser.add_argument("--multiprocessing", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed to use for initialization"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
