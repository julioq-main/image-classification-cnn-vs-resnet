"""
Orchestration layer for training and evaluating image classification models.
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


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training and evaluation pipeline.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with the following attributes:

        - ``config`` : str
            Path to the YAML configuration file. Defaults to
            ``experiments/testing/config.yaml``.
        - ``mode`` : str
            Execution mode. One of ``train`` or ``test``.
        - ``log_level`` : str or None
            Logging level (e.g. ``INFO``, ``DEBUG``). If ``None``,
            falls back to the value in the config file, then to ``INFO``.
        - ``log_file`` : str or None
            Path to a log file. If ``None``, falls back to the value in the
            config file, then logs to stdout only.
        - ``checkpoint`` : str or None
            Path to a checkpoint file. Required when ``mode`` is ``test``.
            Optional when ``mode`` is ``train``; if provided, training
            resumes from that checkpoint.
        - ``history`` : str or None
            Path to a ``history.json`` file from a previous run. Optional;
            ignored if ``checkpoint`` is ``None``.

    Raises
    ------
    SystemExit
        If ``--mode test`` is passed without ``--checkpoint``.
    """
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
        required=False,
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


def main() -> None:
    """
    Entry point for the training and evaluation pipeline.

    Parses command-line arguments, loads the YAML configuration file,
    initialises the logger and optional random seed, then dispatches to
    the appropriate pipeline based on the selected mode:

    - ``train`` : Runs the full training loop via ``run_training``, followed
      by evaluation on the test set via ``run_test``. Training and test
      plots are generated if enabled in the configuration.
    - ``test``  : Runs evaluation only via ``run_test`` using a provided
      checkpoint. Test plots are generated if enabled in the configuration.

    Command-line arguments take precedence over config file values for
    ``log_level`` and ``log_file``. All other settings are read from the
    YAML config.

    Examples
    --------
    Train from scratch:

    >>> python main.py --config experiments/exp001/config.yaml --mode train

    Resume training from a checkpoint:

    >>> python main.py --config experiments/exp001/config.yaml \\
    ...     --checkpoint experiments/exp001/checkpoints/best_model.pth \\
    ...     --history experiments/exp001/history.json

    Evaluate a checkpoint:

    >>> python main.py --mode test \\
    ...     --checkpoint experiments/exp001/checkpoints/best_model.pth
    """
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
            else:
                save_plot_dir = None

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