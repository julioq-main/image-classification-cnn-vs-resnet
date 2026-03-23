import torch
import random
import numpy as np

def set_seed(seed: int):
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(0)

    # Reduce performance
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    
    # deterministic dataloader too?
    