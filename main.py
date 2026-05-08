"""
"""
import yaml
import argparse
import logging
from pathlib import Path

from training import run_training, run_test
from utils.seed import set_seed
from utils.logger import set_logger
from utils.plotting import plot_test, plot_train

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Training and benchmarking models",
    )
    parser.add_argument(
        "--config",
        "-cfg",
        type=str,
        default="experiments/testing/config.yaml",
        required=False,
        help="Path to the YAML config file (e.g. config.yaml)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["test","train"],
        required=False,
        default="train",
        help="Mode for executing: test or train",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        default=None,
        required=False,
        help="Level of the logger",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        required=False,
        help="Path to the logger file",
    )
    parser.add_argument(
        "--checkpoint",
        "-ckpt",
        type=str,
        default=None,
        required=True,
        help="Path to a checkpoint file to resume training from",
    )
    parser.add_argument(
        "--history",
        "-hist",
        type=str,
        default=None,
        required=False,
        help="Path to a history file from a previous run",
    )

    args = parser.parse_args()
    
    if args.mode == "test" and args.checkpoint is None:
        parser.error("--checkpoint is required when mode is 'test'")
    
    return args


def main():
    
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    log_level = args.log_level or cfg.get("log_level","INFO")
    log_file = args.log_file or cfg.get("log_file",None)
    set_logger(log_level=log_level, log_file=log_file)

    seed = cfg.get("seed", None)
    if seed is not None:
        set_seed(seed)
    else:
        logger.info("Seed not set")

    save_dir = cfg.get("save_dir", None)

    if args.mode == "test":
        checkpoint_path = args.checkpoint

        test_metrics, class_names = run_test(cfg, checkpoint_path=checkpoint_path)
    
        if cfg["test"].get("plotting", False):
            if save_dir is not None:
                save_plot_dir = Path(save_dir) / "visualisation" / "test"
        
        plot_test(
            test_metrics,
            class_names=class_names,
            save_dir=save_plot_dir,
        )

    else:
        checkpoint_path = args.checkpoint
        history_path = args.history
        model, history, class_names = run_training(
            cfg,
            checkpoint_path=checkpoint_path,
            history_path=history_path,
        )
        
        train_plot_cfg = cfg["train"].get("plotting", False)
        if train_plot_cfg and train_plot_cfg.get("enabled", False):
            if save_dir is not None:
                save_plot_dir = Path(save_dir) / "visualisation" / "train"
            else:
                save_plot_dir = None
                
            plot_train(
                history,
                cfg=train_plot_cfg,
                class_names=class_names,
                save_dir=save_plot_dir,
            )

        test_metrics, class_names = run_test(cfg, model=model)
        test_plot_cfg = cfg["test"].get("plotting", False)
        if test_plot_cfg and test_plot_cfg.get("enabled", False):
            if save_dir is not None:
                save_plot_dir = Path(save_dir) / "visualisation" / "test"
            else:
                save_plot_dir = None
                
            plot_test(
                test_metrics,
                class_names=class_names,
                save_dir=save_plot_dir,
            )
    

if __name__ == "__main__":
    main()