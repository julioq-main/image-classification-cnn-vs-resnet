"""
TODO try dict instead  of match pattern? (Search which is better)
TODO doc

"""

import torch.optim as optim

def get_optim(cfg, model):    
    
    # --- Optimizer ---
    match cfg["name"]:
        case "sgd":
            optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])
        case "adam":
            optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        case "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"])
        case _:
            raise ValueError("Unknown optimizer")
        
    return optimizer