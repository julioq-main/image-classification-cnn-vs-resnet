"""

"""

import yaml
import argparse
from training.train import run_training
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(prog="Parser training and benchmarking models",)
    parser.add_argument("--config", "-c", type=str, default="experiments/testing/config.yaml", required=False, help="Path to the YAML config file (e.g. config.yaml)")
    #Add more arguments and implement logic
    parser.add_argument("--save-dir", "-sd", type=str, required=False, help="Path to the directory to save")
    parser.add_argument("--training", "-t", type=bool, required=False, default=True)
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", None)
    if seed is not None:
        print("Seed set")
        set_seed(seed)

    model, history = run_training(config)


if __name__ == "__main__":
    main()