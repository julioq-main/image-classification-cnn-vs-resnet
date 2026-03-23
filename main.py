"""
TODO Plot loss and accuracy during training, maybe add a bool parameter to activate the plotting function
TODO improve everything
TODO doc
TODO set seed (utils/seed.py)

"""



import yaml
from training.train import run_training
from utils.seed import set_seed

def main():
    
    with open("experiments/test/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", None)
    if seed is not None:
        set_seed(seed)

    model, history = run_training(config)


if __name__ == "__main__":
    main()