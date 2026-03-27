import torch
import random
import numpy as np

def set_seed(seed: int):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Reduce performance
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    
    # deterministic dataloader too?
    