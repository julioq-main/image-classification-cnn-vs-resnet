"""
Training pipeline for image classification models.

This module exposes a single entry point, `run_training`, which takes a
configuration dictionary and executes a full training loop with optional
checkpointing, early stopping, and advanced evaluation metrics.
"""
import json
import logging
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn

from models import get_model
from utils.data import get_dataloader
from utils.optim import get_optim
from utils.metrics import compute_advanced_metrics
from engine import train_one_epoch, eval_one_epoch

logger = logging.getLogger(__name__)


def run_training(cfg: dict, checkpoint_path: str | Path | None = None) -> tuple[nn.Module, dict]:
    """
    Execute the full training loop for an image classification model.

    Trains a model for a fixed number of epochs with optional early stopping,
    loss-goal termination, and best-model checkpointing. If a save directory
    is provided, the best checkpoint and training history are persisted to
    disk; otherwise the best model weights are kept in memory.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary. Expected top-level keys:

        - ``seed`` : int, optional
            Global random seed passed to the dataloader.
        - ``save_dir`` : str or path-like, optional
            Root directory for checkpoints and history. If omitted, the best
            model state is kept in memory and no files are written.
        - ``model`` : dict
            Passed directly to ``get_model``. Must contain at least ``name``
            and ``num_classes``.
        - ``data`` : dict
            Passed directly to ``get_dataloader``. Must contain dataset paths,
            normalization stats, and loader settings.
        - ``training`` : dict
            Training settings. Expected keys:

            - ``optimizer`` : dict
                Passed to ``get_optim``. Must contain ``name`` and ``lr``.
            - ``epochs`` : int
                Maximum number of training epochs.
            - ``patience`` : int, optional
                Early stopping patience. Training stops when validation loss
                has not improved for this many consecutive epochs.
            - ``loss_goal`` : float, optional
                Training stops immediately when validation loss drops below
                this threshold.
        - ``eval`` : dict
            Evaluation settings. Expected keys:

            - ``advanced_metrics`` : bool, default=False
                If ``True``, per-epoch macro-averaged precision, recall and
                F1 and confusion matrix are computed and stored in the
                returned history.
    checkpoint_path : str or path-like or None, optional
        Path to a checkpoint file to resume training from. If provided, the
        model weights, optimizer state, best validation loss, and starting
        epoch are restored from the checkpoint before training begins.
        If ``None``, training starts from scratch.
                
    Returns
    -------
    model : torch.nn.Module
        The model loaded with the best weights observed during training
        (lowest validation loss).
    history : dict
        Training history with the following keys, each being a list of
        per-epoch values:

        - ``train_loss`` : list of float
        - ``train_accuracy`` : list of float
        - ``val_loss`` : list of float
        - ``val_accuracy`` : list of float

        When ``advanced_metrics`` is enabled, the following keys are also
        present:

        - ``macro_precision`` : list of float
        - ``macro_recall`` : list of float
        - ``macro_f1`` : list of float
        - ``confusion_matrix`` : list of list of int
            One matrix per epoch.

    Notes
    -----
    The loss criterion is fixed to `torch.nn.CrossEntropyLoss` as this
    pipeline is designed exclusively for multi-class classification tasks.

    When ``save_dir`` is provided, the best checkpoint is saved to
    ``<save_dir>/checkpoints/checkpoint.pth`` and the final model weights to
    ``<save_dir>/checkpoints/final_model.pth``. Training history is written
    to ``<save_dir>/history.json``. Non-serializable values (e.g. tensors)
    are converted via ``.tolist()`` automatically.

    Examples
    --------
    >>> with open("config.yaml") as f:
    ...     cfg = yaml.safe_load(f)
    >>> model, history = run_training(cfg)
    >>> print(history["val_loss"])
    """
    logger.info("Starting training")
    
    epochs = cfg["training"]["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model=get_model(cfg["model"]).to(device)
    optimizer = get_optim(cfg["training"]["optimizer"], model)
    seed = cfg.get("seed", None)
    loaders = get_dataloader(cfg["data"], seed)
    criterion = nn.CrossEntropyLoss()

    use_advanced_metrics = cfg["eval"].get("advanced_metrics", False)
    patience = cfg["training"].get("patience", None)
    patience_counter = 0
    loss_goal = cfg["training"].get("loss_goal", None)
    best_val_loss = float("inf")
    best_model = None  # Holds in-memory best weights when save_dir is None

    #Resume from checkpoint if provided
    if checkpoint_path is not None:
        checkpoint = torch.load(
            save_checkpoint_path,
            weights_only=True,
            map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["val_loss"]
        logger.info(
            f"Resuming from checkpoint at epoch {start_epoch}, "
            f"val_loss {best_val_loss:.4f}"
        )
    else:
        start_epoch = 0

    save_dir = cfg.get("save_dir", None)
    if save_dir is not None:
        logger.info("Saving to disk is active")
        save_dir = Path(save_dir)
        save_checkpoint_dir = save_dir / "checkpoints"
        save_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint_path = save_checkpoint_dir / "checkpoint.pth"
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
            "macro_precision":[],
            "macro_recall": [],
            "macro_f1": [],
            "confusion_matrix": []
        })
        
    for epoch in range(start_epoch, epochs):
        model.train()
        train_metrics = train_one_epoch(
            loaders["train_loader"],
            model,
            criterion,
            optimizer,
            device,
        )
        model.eval()
        val_metrics = eval_one_epoch(loaders["val_loader"], model, criterion, device)
        
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        logger.info(
            f"Epoch [{epoch+1}/{epochs}]  "
            f"Train — loss: {train_metrics['loss']:.4f}  acc: {train_metrics['accuracy']:.4f}  |  "
            f"Val — loss: {val_metrics['loss']:.4f}  acc: {val_metrics['accuracy']:.4f}"
        )
        
        if use_advanced_metrics:
            advanced_metrics = compute_advanced_metrics(
                val_metrics["targets"],
                val_metrics["preds"],
            )
            
            history["macro_precision"].append(advanced_metrics["macro_precision"])
            history["macro_recall"].append(advanced_metrics["macro_recall"])
            history["macro_f1"].append(advanced_metrics["macro_f1"])
            history["confusion_matrix"].append(advanced_metrics["confusion_matrix"])

            logger.info(
                f"Epoch [{epoch+1}/{epochs}]  "
                f"Adv — precision: {advanced_metrics['macro_precision']:.4f}  "
                f"recall: {advanced_metrics['macro_recall']:.4f}  "
                f"f1: {advanced_metrics['macro_f1']:.4f}"
            )

        #Saving best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            
            if save_dir is not None:
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                }, save_checkpoint_path)
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
            logger.info(
                f"Early stopping at epoch {epoch+1} — no improvement for {patience} epochs"
            )
            break
    
    # Restore best weights and persist outputs
    if save_dir is not None:
        checkpoint = torch.load(
            save_checkpoint_path,
            weights_only=True,
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        
        final_path = Path(save_checkpoint_dir) / "final_model.pth"
        torch.save(model.state_dict(), final_path)

        history_path = save_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=1, default=lambda x: x.tolist())

        logger.info(f"Final model and history saved to {save_checkpoint_dir}")
    else:
        if best_model is not None:
            model.load_state_dict(best_model)
    
    logger.info("Training complete")
    
    return model, history