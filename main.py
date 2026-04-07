"""

"""

import yaml
import argparse
import logging
from training.train import run_training
from utils.seed import set_seed
from utils.logger import set_logger

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(prog="Parser training and benchmarking models",)
    parser.add_argument("--config",
                        "-c",
                        type=str,
                        default="experiments/testing/config.yaml",
                        required=False,
                        help="Path to the YAML config file (e.g. config.yaml)"
                        )
    
    #Add more arguments and implement logic
    parser.add_argument("--log-level",
                        "-l", type=str,
                        default=None,
                        required=False,
                        help="Level of the logger"
                        )
    
    parser.add_argument("--log-file", type=str, default=None, help="Path to the logger file")
    parser.add_argument("--save-dir", "-sd", type=str, required=False, help="Path to the directory to save")
    parser.add_argument("--training", "-t", type=bool, required=False, default=True)
    
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
        logger.info(f"Seed set to {seed}")
        set_seed(seed)
    else:
        logger.info("Seed not set")

    model, history = run_training(config)


if __name__ == "__main__":
    main()