"""
TODO doc

TODO logging? Example below (AI made):

# utils/logging.py

import logging
import sys

def setup_logger(log_level="INFO", log_file=None):
    logger = logging.getLogger("training")
    logger.setLevel(getattr(logging, log_level.upper()))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# main.py

logger = setup_logger(log_level="INFO", log_file="train.log")
run_training(cfg, logger)

# training/train.py

logger.info("Starting training")

logger.info(f"Epoch {epoch+1}/{epochs}")

logger.info(
    "Train | loss: %.4f | acc: %.4f",
    train_metrics["loss"],
    train_metrics["accuracy"]
)

logger.info(
    "Val   | loss: %.4f | acc: %.4f",
    val_metrics["loss"],
    val_metrics["accuracy"]
)

if cfg["eval"].get("advanced_metrics", False):
    logger.debug(
        "Advanced | precision: %.4f | recall: %.4f | f1: %.4f",
        advanced_metrics["precision"],
        advanced_metrics["recall"],
        advanced_metrics["f1_score"]
    )

if loss_goal is not None and val_metrics["loss"] < loss_goal:
    logger.info("Loss goal reached: %.4f", val_metrics["loss"])
    break

if patience is not None and patience_counter >= patience:
    logger.info("Early stopping triggered (patience=%d)", patience)
    break

    
Need to learn how it works and decide if implementing and removing print statements


"""

import torch
import torch.nn as nn
from copy import deepcopy

from models import get_model
from utils.data import get_dataloader
from utils.optim import get_optim
from utils.metrics import compute_metrics
from engine.engine import train_one_epoch, eval_one_epoch


def run_training(cfg):
    
    # --- Epochs & Device ---
    epochs = cfg["training"]["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Model ---
    model=get_model(cfg["model"]).to(device)

    # --- Optimizer ---
    optimizer = get_optim(cfg["training"]["optimizer"], model)

    # --- DataLoaders
    loaders = get_dataloader(cfg)

    # --- Criterion ---
    criterion = nn.CrossEntropyLoss().to(device)   #Always same criterion as it is a classification task

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
            advanced_metrics = compute_metrics(val_metrics["targets"],val_metrics["preds"])
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
