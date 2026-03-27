"""

"""

import yaml
from training.train import run_training
from utils.seed import set_seed

def main():
    
    with open("experiments/test/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", None)
    if seed is not None:
        print("Seed set")
        set_seed(seed)

    model, history = run_training(config)


if __name__ == "__main__":
    main()