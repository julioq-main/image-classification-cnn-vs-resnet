"""
TODO add other models
TODO doc
"""


from models.models import MyResNet18

def get_model(cfg):
    
    name = cfg["name"]
    num_classes = cfg["num_classes"]

    # --- Model ---
    match name:
        case "resnet":
            model = MyResNet18(num_classes)
        #TODO add other models
        case _:
            raise ValueError("Unknown model")
        
    return model