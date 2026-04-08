"""


"""
import json
import logging
import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy

from models import get_model
from utils.data import get_dataloader
from utils.optim import get_optim
from utils.metrics import compute_metrics
from engine import train_one_epoch, eval_one_epoch

logger = logging.getLogger(__name__)

def run_training(cfg):
    logger.info("Starting training")
    
    epochs = cfg["training"]["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model=get_model(cfg["model"]).to(device)
    optimizer = get_optim(cfg["training"]["optimizer"], model)
    loaders = get_dataloader(cfg["data"])
    criterion = nn.CrossEntropyLoss()   #Always same criterion as it is a classification task

    use_advanced_metrics = cfg["eval"].get("advanced_metrics", False)
    patience = cfg["training"].get("patience", None)
    patience_counter = 0
    loss_goal = cfg["training"].get("loss_goal", None)
    best_val_loss = float("inf")

    save_dir = cfg.get("save_dir", None)
    if save_dir is not None:
        logger.info("Saving to disk is active")
        save_dir = Path(save_dir)
        checkpoint_dir = save_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint.pth"
    else:
        best_model = deepcopy(model.state_dict())
        logger.info("Saving to disk is not active")
        
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        }

    if use_advanced_metrics:
        logger.info("Advanced metrics are active")
        history.update({
            "precision":[],
            "recall": [],
            "f1_score": [],
            "confusion_matrix": []
        })
        
    for epoch in range(epochs):
        model.train()
        train_metrics = train_one_epoch(loaders["train_loader"], model, criterion, optimizer, device)
        model.eval()
        val_metrics = eval_one_epoch(loaders["val_loader"], model, criterion, device)
        
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        logger.info(f"Epoch [{epoch+1}/{epochs}]  "
            f"Train — loss: {train_metrics['loss']:.4f}  acc: {train_metrics['accuracy']:.4f}  |  "
            f"Val — loss: {val_metrics['loss']:.4f}  acc: {val_metrics['accuracy']:.4f}")

        
        if use_advanced_metrics:
            advanced_metrics = compute_metrics(val_metrics["targets"], val_metrics["preds"])
            
            history["precision"].append(advanced_metrics["precision"])
            history["recall"].append(advanced_metrics["recall"])
            history["f1_score"].append(advanced_metrics["f1_score"])
            history["confusion_matrix"].append(advanced_metrics["confusion_matrix"])

            logger.info(f"Epoch [{epoch+1}/{epochs}]  "
            f"Adv — precision: {advanced_metrics['precision']:.4f}  recall: {advanced_metrics['recall']:.4f}  f1: {advanced_metrics['f1_score']:.4f}")

        #Saving best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            
            if save_dir is not None:
                torch.save({
                    "epoch": epoch+1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                }, checkpoint_path)
            else:
                best_model = deepcopy(model.state_dict())

            logger.info(f"Checkpoint saved — val_loss improved to {best_val_loss:.4f}")
            
            patience_counter = 0
        else:
            patience_counter +=1

        # Conditional stops
        if loss_goal is not None and val_metrics["loss"] < loss_goal:
            logger.info(f"Loss goal ({loss_goal}) reached at epoch {epoch+1}")
            break
        elif patience is not None and patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} — no improvement for {patience} epochs")
            break

    if save_dir is not None:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        final_path = Path(checkpoint_dir) / "final_model.pth"
        torch.save(model.state_dict(), final_path)

        history_path = save_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))

        logger.info(f"Final model and history saved to {final_path}")
    else:
        model.load_state_dict(best_model)
    
    logger.info("Training complete")
    
    return model, history
