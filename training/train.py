"""


"""

import torch
import torch.nn as nn
from copy import deepcopy

from models import get_model
from utils.data import get_dataloader
from utils.optim import get_optim
from utils.metrics import compute_metrics
from engine import train_one_epoch, eval_one_epoch


def run_training(cfg):
    
    # --- Epochs & Device ---
    epochs = cfg["training"]["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Model ---
    model=get_model(cfg["model"]).to(device)

    # --- Optimizer ---
    optimizer = get_optim(cfg["training"]["optimizer"], model)

    # --- DataLoaders
    loaders = get_dataloader(cfg["data"])

    # --- Criterion ---
    criterion = nn.CrossEntropyLoss()   #Always same criterion as it is a classification task

    # --- History ---
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "precision":[],
        "recall": [],
        "f1_score": [],
        "confusion_matrix": []
    }

    # Early stopping variables
    patience = cfg["training"].get("patience", None)
    patience_counter = 0
    best_model = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    loss_goal = cfg["training"].get("loss_goal",None)

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

        if cfg["eval"].get("advanced_metrics", False):
            advanced_metrics = compute_metrics(val_metrics["targets"], val_metrics["preds"])
            print("   --- Advanced metrics --- ")
            print(f"   Precision: {advanced_metrics["precision"]};  Recall: {advanced_metrics["recall"]};  F1 Score: {advanced_metrics["f1_score"]}\n")

            history["precision"].append(advanced_metrics["precision"])
            history["recall"].append(advanced_metrics["recall"])
            history["f1_score"].append(advanced_metrics["f1_score"])
            history["confusion_matrix"].append(advanced_metrics["confusion_matrix"])

        #Saving best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter +=1

        # Conditional stops
        if loss_goal is not None and val_metrics["loss"] < loss_goal:    
            print("Loss goal has been reached")
            break
        
        if patience is not None and patience_counter >= patience:
            print("Training has plateaued")
            break

    model.load_state_dict(best_model)
    
    print("|======================================|")
    print("|--------- TRAINING COMPLETED ---------|")
    print("|======================================|")

    return model, history
