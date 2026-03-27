"""


"""

import torch.optim as optim

def get_optim(cfg, model):    
    
    name = cfg.get("name", "No name added")
    # --- Optimizer ---
    match name:
        case "sgd":
            lr = cfg.get("lr", 0.001)
            momentum = cfg.get("momentum", 0.9)
            weight_decay=cfg.get("weight_decay", 0.0)

            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        case "adam":
            lr = cfg.get("lr", 0.001)
            weight_decay=cfg.get("weight_decay", 0.0)

            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        case "adamw":
            lr = cfg.get("lr", 0.001)
            weight_decay=cfg.get("weight_decay", 0.01)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        case _:
            raise ValueError(f"Unknown optimizer: '{name}'")
        
    return optimizer