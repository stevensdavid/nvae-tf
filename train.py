# import tensorflow as tf
# tf.config.experimental.set_visible_devices([], "GPU")
from models import NVAE
from datasets import load_mnist
from argparse import ArgumentParser

def main(args):
    train_data, test_data = load_mnist(batch_size=8)
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
        scale_factor=args.scale_factor
    )
    model.compile(optimizer="adamax", run_eagerly=True)
    callbacks = [

    ]
    # for batch_x, _ in train_data:
    #     model(batch_x)
    model.fit(train_data, epochs=10, )

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n_encoder_channels", type=int, help="")
    parser.add_argument("--n_decoder_channels", type=int, help="")
    parser.add_argument("--res_cells_per_group", type=int, help="")
    parser.add_argument("--n_preprocess_blocks", type=int, help="")
    parser.add_argument("--n_preprocess_cells", type=int, help="")
    parser.add_argument("--n_postprocess_blocks", type=int, help="")
    parser.add_argument("--n_postprocess_cells", type=int, help="")
    parser.add_argument("--n_latent_per_group", type=int, help="")
    parser.add_argument("--n_groups_per_scale", type=int, help="")
    parser.add_argument("--sr_lambda", type=int, help="")
    parser.add_argument("--scale_factor", type=int, default=2, help="")

if __name__ == "__main__":
    
    pass
