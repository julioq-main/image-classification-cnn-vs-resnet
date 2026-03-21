"""
TODO Plot loss and accuracy during training, maybe add a bool parameter to activate the plotting function
TODO Make it a function to call
"""

import yaml
import torch
import torch.optim as optim
import torch.nn as nn

from utils.data import get_dataloader
from engine.engine import train_one_epoch, eval_one_epoch
from engine.metrics import compute_metrics
from models.models import MyResNet18

#Loading config file

def run_training(cfg):
    
    # --- Epochs & Device ---
    epochs = cfg["training"]["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Model ---
    match cfg["model"]:
        case "resnet":
            model = MyResNet18(25)
        #TODO add other models
        case _:
            raise ValueError("Unknown model")
    model.to(device)


    # --- Optimizer ---
    match cfg["training"]["optimizer"]:
        case "sgd":
            optimizer = optim.SGD(model.parameters(), lr=cfg["training"]["lr"], momentum=cfg["training"]["momentum"])
        case "adam":
            optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
        case "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
        case _:
            raise ValueError("Unknown optimizer")

    # --- DataLoaders
    loaders = get_dataloader(cfg)

    # --- Criterion ---
    criterion = nn.CrossEntropyLoss()   #Always same criterion as it is a classification task
    criterion.to(device)

    # --- History ---
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "precision_score":[],
        "recall_score": [],
        "f1_score": [],
        "confusion_matrix": []
    }

    # --- Training ---
    print("|=====================================|")
    print("|--------- STARTING TRAINING ---------|")
    print("|=====================================|\n")

    for epoch in range(epochs):
        print(f"=========== Epoch {epoch+1} ===========")
        
        train_metrics = train_one_epoch(loaders["train_loader"], model, criterion, optimizer, device)
        print(f"   --- Training metrics ---")
        print(f"   Average loss: {train_metrics["loss"]};  Accuracy: {train_metrics["accuracy"]}\n")
        
        val_metrics = eval_one_epoch(loaders["val_loader"], model, criterion, device)
        print(f"   --- Validating metrics ---")
        print(f"   Average loss: {val_metrics["loss"]};  Accuracy: {val_metrics["accuracy"]}\n")

        # --- Store metrics ---
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        if cfg["eval"]["advanced_metrics"]:
            advanced_metrics = compute_metrics(val_metrics["targets"],val_metrics["preds"])
            print("   --- Advanced metrics --- ")
            print(f"   Precision Score: {advanced_metrics["precision_score"]};  Recall Score: {advanced_metrics["recall_score"]};  F1 Score: {advanced_metrics["f1_score"]}\n")

            history["precision_score"].append(advanced_metrics["precision_score"])
            history["recall_score"].append(advanced_metrics["recall_score"])
            history["f1_score"].append(advanced_metrics["f1_score"])
            history["confusion_matrix"].append(advanced_metrics["confusion_matrix"])
    
    print("|======================================|")
    print("|--------- TRAINING COMPLETED ---------|")
    print("|======================================|")
    return model, history




def main():
    #TODO Write better main function
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    run_training(config)

if __name__ == "__main__":
    main()