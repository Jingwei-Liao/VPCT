import argparse

from compressai.zoo import image_models


CUSTOM_MODELS = ["vpct-cheng2020-attn"]


def parse_args(argv):
    parser = argparse.ArgumentParser(description="360 ERP viewport training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=sorted(set(image_models.keys()) | set(CUSTOM_MODELS)),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset")
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Subdirectory name for train split",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="test",
        help="Subdirectory name for test split",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Number of batches prefetched by each worker (default: %(default)s)",
    )
    parser.add_argument(
        "--no-persistent-workers",
        action="store_true",
        help="Disable persistent dataloader workers between epochs",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=None,
        help="Bit-rate distortion parameter; default is auto-selected from quality",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--vp-fov",
        type=float,
        nargs=2,
        default=(90.0, 90.0),
        help="Viewport FoV as (vertical horizontal); output size is auto-adapted from ERP resolution",
    )
    parser.add_argument(
        "--num-viewports",
        type=int,
        default=6,
        help="Number of viewports sampled per ERP image",
    )
    parser.add_argument(
        "--vpct-layers",
        type=int,
        default=1,
        help="Number of stacked VPCT layers (Intra+Inter pairs)",
    )
    parser.add_argument(
        "--random-vp-rotate",
        action="store_true",
        help="Enable random yaw rotation for viewport extraction in training",
    )
    parser.add_argument(
        "--random-vp-subset",
        action="store_true",
        help="Sample random subset of viewports for each training sample",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=3,
        help="CompressAI model quality index",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s)",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument(
        "--save-root",
        type=str,
        default="checkpoints",
        help="Root directory for saving experiment outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom experiment subdirectory name; auto-generated when omitted",
    )
    return parser.parse_args(argv)
