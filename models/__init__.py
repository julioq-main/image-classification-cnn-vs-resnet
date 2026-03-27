"""
TODO add other models
TODO doc
"""


from models.architectures import MyResNet18

def get_model(cfg):
    
    name = cfg["name"]
    num_classes = cfg["num_classes"]

    # --- Model ---
    match name:
        case "resnet":
            model = MyResNet18(num_classes)
        #TODO add other models
        case _:
            raise ValueError(f"Unknown model: '{name}'")
        
    return model