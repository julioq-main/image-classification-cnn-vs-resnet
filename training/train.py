"""


"""
import logging
import torch
import torch.nn as nn
from copy import deepcopy

from models import get_model
from utils.data import get_dataloader
from utils.optim import get_optim
from utils.metrics import compute_metrics
from engine import train_one_epoch, eval_one_epoch

logger = logging.getLogger(__name__)

def run_training(cfg):

    logger.info("|--------- STARTING TRAINING ---------|")
    
    epochs = cfg["training"]["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=get_model(cfg["model"]).to(device)
    optimizer = get_optim(cfg["training"]["optimizer"], model)
    loaders = get_dataloader(cfg["data"])
    criterion = nn.CrossEntropyLoss()   #Always same criterion as it is a classification task

    use_advanced_metrics = cfg["eval"].get("advanced_metrics", False)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        
    }

    if use_advanced_metrics:
        logger.debug("Advanced Metrics are active")
        history.update({
            "precision":[],
            "recall": [],
            "f1_score": [],
            "confusion_matrix": []
            })
        
    # Early stopping variables
    patience = cfg["training"].get("patience", None)
    patience_counter = 0
    best_model = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    loss_goal = cfg["training"].get("loss_goal",None)


    for epoch in range(epochs):
        logger.info(f"=========== Epoch {epoch+1} ===========")
        
        train_metrics = train_one_epoch(loaders["train_loader"], model, criterion, optimizer, device)
        logger.info(f"Training metrics:  Average loss={train_metrics["loss"]};  Accuracy={train_metrics["accuracy"]}")
        
        val_metrics = eval_one_epoch(loaders["val_loader"], model, criterion, device)
        logger.info(f"Validation metrics:  Average loss: {val_metrics["loss"]};  Accuracy: {val_metrics["accuracy"]}\n")

        # --- Store metrics ---
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        if use_advanced_metrics:
            
            advanced_metrics = compute_metrics(val_metrics["targets"], val_metrics["preds"])

            logger.info(f"Advanced metrics enabled:  Precision: {advanced_metrics["precision"]};  Recall: {advanced_metrics["recall"]};  F1 Score: {advanced_metrics["f1_score"]}")

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
            logger.info("Loss goal has been reached")
            break
        
        if patience is not None and patience_counter >= patience:
            logger.info("Training has plateaued")
            break

    model.load_state_dict(best_model)
    
    logger.info("|--------- TRAINING COMPLETED ---------|")
    
    return model, history
