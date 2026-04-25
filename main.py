"""

"""

import yaml
import argparse
import logging

from training.train import run_training
from utils.seed import set_seed
from utils.logger import set_logger
from utils.plotting import plot_accuracy, plot_loss

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(prog="Parser training and benchmarking models",)
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="experiments/testing/config.yaml",
        required=False,
        help="Path to the YAML config file (e.g. config.yaml)"
    )
    
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        default=None,
        required=False,
        help="Level of the logger"
    )
    
    parser.add_argument(
        "--checkpoint",
        "-ckpt",
        type=str,
        default=None,
        required=False,
        help="Path to a checkpoint file to resume training from"
    )

    parser.add_argument("--log-file", type=str, default=None, help="Path to the logger file")
    
    return parser.parse_args()


def main():
    
    args = parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    log_level = args.log_level or config.get("log_level","INFO")
    log_file = args.log_file or config.get("log_file",None)

    set_logger(log_level=log_level, log_file=log_file)

    seed = config.get("seed", None)
    if seed is not None:
        set_seed(seed)
    else:
        logger.info("Seed not set")

    save_dir = args.save_dir or config.get("save_dir", None)

    model, history = run_training(config)

    train_loss = history["train_loss"]
    train_acc = history["train_accuracy"]
    val_loss = history["val_loss"]
    val_acc = history["val_accuracy"]

    if save_dir is not None:
        loss_save_path = save_dir + "/visualisation/loss"
        acc_save_path = save_dir + "/visualisation/acc"
    else:
        loss_save_path = None
        acc_save_path = None
        
    plot_loss(train_loss=train_loss, val_loss=val_loss, save_path=loss_save_path)
    plot_accuracy(train_acc=train_acc, val_acc=val_acc, save_path=acc_save_path)


if __name__ == "__main__":
    main()