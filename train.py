import sys

from training.args import parse_args
from training.runner import run_training


if __name__ == "__main__":
    run_training(parse_args(sys.argv[1:]))
