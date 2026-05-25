"""
Training pipeline for image classification models.
"""
import json
import logging
from pathlib import Path
from copy import deepcopy
import time

import torch
import torch.nn as nn

from models import get_model
from utils.data import get_dataloader
from utils.optim import get_optim, get_scheduler
from utils.metrics import compute_advanced_metrics
from engine import train_one_epoch, eval_one_epoch

logger = logging.getLogger(__name__)


def run_training(
        cfg: dict,
        checkpoint_path: str | Path | None = None,
        history_path: str | Path | None = None,
    ) -> tuple[nn.Module, dict]:
    """
    Execute the full training loop for an image classification model.

    Trains a model for a fixed number of epochs with optional early stopping,
    loss-goal termination, and best-model checkpointing. If a save directory
    is provided, the best checkpoint and training history are saved to disk;
    otherwise the best model weights are only returned.

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
        - ``train`` : dict
            Training settings. Expected keys:

            - ``checkpoint_interval`` : int
                Interval of epochs between saving a checkpoint.
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
    checkpoint_path : str or Path or None, optional
        Path to a checkpoint file to resume training from. If provided, the
        model weights, optimizer state, best validation loss, and starting
        epoch are restored from the checkpoint before training begins.
        If ``None``, training starts from scratch.
    history_path : str or Path or None, optional
        Path to a ``history.json`` file from a previous run, used to restore
        training history when resuming from a checkpoint. If provided, the
        history is truncated to ``checkpoint_path``'s epoch so it aligns with
        the resumed model state. Ignored if ``checkpoint_path`` is ``None``.
        If ``None``, history will only cover epochs from the current run.
                
    Returns
    -------
    model : torch.nn.Module
        The model loaded with the best weights observed during training
        (lowest validation loss).
    history : dict
        Training history with the following keys, each being a list of
        per-epoch values:

        - ``epoch``: list of int
            Epoch numbers corresponding to each entry.
        - ``duration``: list of float
            Duration of each epoch measured in seconds.
        - ``train_loss`` : list of float
            Average training loss across all samples for each epoch.
        - ``train_accuracy`` : list of float
            Average training accuracy across all samples for each epoch.
        - ``val_loss`` : list of float
            Average validation loss across all samples for each epoch.
        - ``val_accuracy`` : list of float
            Average validation accuracy across all samples for each epoch.

        When ``advanced_metrics`` is enabled, the following keys are also
        present:

        - ``macro_precision`` : list of float
            Macro-averaged precision across all classes for each epoch.
        - ``macro_recall`` : list of float
            Macro-averaged recall across all classes for each epoch.
        - ``macro_f1`` : list of float
            Macro-averaged F1-score across all classes for each epoch.
        - ``confusion_matrix`` : np.ndarray
            Confusion matrix of shape ``(C, C)``, where ``C`` is the number of
            classes. One matrix per epoch.
    class_names: list of str
        Labels for each class.
            
    Notes
    -----
    The loss criterion is fixed to ``torch.nn.CrossEntropyLoss`` as this
    pipeline is designed exclusively for multi-class classification tasks.

    When ``save_dir`` is provided, in every ``checkpoint_interval`` the model
    is saved to ``<save_dir>/checkpoints/checkpoint_epoch_{epoch}``, the best
    checkpoint is saved to ``<save_dir>/checkpoints/best_model.pth`` and the
    final model weights to ``<save_dir>/checkpoints/last_model.pth``. Training
    history is written to ``<save_dir>/history.json``.

    If ``history_path`` is provided, the loaded history keys must match those
    of the current run. If they differ (e.g. ``advanced_metrics`` was toggled
    between runs), the loaded history is discarded and a warning is logged.
    If ``history_path`` is provided without ``checkpoint_path``, it is ignored
    and a warning is logged.

    Examples
    --------
    >>> with open("config.yaml") as f:
    ...     cfg = yaml.safe_load(f)
    >>> model, history = run_training(cfg)
    >>> print(history["val_loss"])

    >>> # Resume training from a previous checkpoint
    >>> model, history = run_training(
    ...     cfg,
    ...     checkpoint_path="experiments/exp001/checkpoints/best_model.pth",
    ...     history_path="experiments/exp001/history.json"
    ...     )
    """
    logger.info("Starting training")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    epochs = cfg["train"]["epochs"]
    model = get_model(cfg["model"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optim(cfg["train"]["optimizer"], model)
    scheduler = get_scheduler(
        cfg["train"]["optimizer"].get("scheduler", None),
        optimizer=optimizer,
    )
    loaders = get_dataloader(cfg["data"], cfg.get("seed", None))
    class_names = loaders["train_loader"].dataset.classes

    use_advanced_metrics = cfg["train"].get("advanced_metrics", False)
    history = {
        "epoch": [],
        "duration": [],
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
            "confusion_matrix": [],
        })

    best_model = deepcopy(model.state_dict())
    patience = cfg["train"].get("patience", None)
    patience_counter = 0
    checkpoint_interval = cfg["train"].get("checkpoint_interval", 10)
    loss_goal = cfg["train"].get("loss_goal", None)
    start_epoch = 0
    best_val_loss = float("inf")

    #Resume from checkpoint if provided
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path,
            weights_only=True,
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                logger.warning(
                    "LR Scheduler is configured but no 'scheduler_state_dict' "
                    "found in checkpoint. Scheduler will start from scratch."
                )
                
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["val_loss"]
        patience_counter = checkpoint["patience_counter"]
        
        logger.info(
            f"Resuming from checkpoint at epoch {start_epoch}, next epoch "
            f"will be {start_epoch+1} with val_loss {best_val_loss:.4f}."
        )

        if history_path is not None:
            with open(history_path, "r") as f:
                checkpoint_history = json.load(f)
            checkpoint_keys = set(checkpoint_history.keys())
            current_keys = set(history.keys())
            if checkpoint_keys != current_keys:
                logger.warning(
                    "History key mismatch between checkpoint history and current run "
                    f"(loaded from checkpoint: {checkpoint_keys}, expected: {current_keys}). "
                    "Discarding history from checkpoint."
                    )
            else:
                history = {k:v[:start_epoch] for k,v in checkpoint_history.items()}
                logger.info("History loaded from checkpoint.")
        else:
            logger.info(
                f"No history_path given. History will start at epoch {start_epoch+1}"
                )        
    elif history_path is not None:
        logger.warning(
        "history_path provided but no checkpoint_path given. "
        "history_path will be ignored."
        )
    
    save_dir = cfg.get("save_dir", None)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_checkpoint_dir = save_dir / "checkpoints"
        save_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving to disk is active")
    else:
        logger.info("Saving to disk is not active")

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        
        model.train()
        train_metrics = train_one_epoch(
            loaders["train_loader"],
            model,
            criterion,
            optimizer,
            device,
        )
        model.eval()
        with torch.no_grad():
            val_metrics = eval_one_epoch(loaders["val_loader"], model, criterion, device)
        
        if scheduler is not None:
            scheduler.step()

        duration = time.time() - start_time

        history["epoch"].append(epoch+1)
        history["duration"].append(duration)
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
            history["confusion_matrix"].append(advanced_metrics["confusion_matrix"].tolist())

            logger.info(
                f"Epoch [{epoch+1}/{epochs}]  "
                f"Adv — precision: {advanced_metrics['macro_precision']:.4f}  "
                f"recall: {advanced_metrics['macro_recall']:.4f}  "
                f"f1: {advanced_metrics['macro_f1']:.4f}"
            )

        if save_dir is not None and (epoch+1) % checkpoint_interval == 0:
            save_checkpoint_path = save_checkpoint_dir / f"checkpoint_epoch_{epoch+1}"
            checkpoint_data = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "patience_counter": patience_counter,
            }
            if scheduler is not None:
                checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

            torch.save(checkpoint_data, save_checkpoint_path)
            
            logger.info(f"Checkpoint saved at epoch {epoch+1}")

        #Saving best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            
            if save_dir is not None:
                save_best_path = save_checkpoint_dir / "best_model.pth"
                
                checkpoint_data = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "patience_counter": patience_counter,                    
                }
                if scheduler is not None:
                    checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
                
                torch.save(checkpoint_data, save_best_path)
            
            best_model = deepcopy(model.state_dict())
            logger.info(f"Checkpoint saved — val_loss improved to {best_val_loss:.4f}")
            
            patience_counter = 0
        else:
            patience_counter += 1

        # Conditional stops
        if loss_goal is not None and val_metrics["loss"] < loss_goal:
            logger.info(f"Loss goal ({loss_goal}) reached at epoch {epoch+1}")
            break
        elif patience is not None and patience_counter >= patience:
            logger.info(
                f"Early stopping at epoch {epoch+1} — no improvement for {patience} epochs"
            )
            break
    
    if save_dir is not None:
        save_last_path = Path(save_checkpoint_dir) / "last_model.pth"
        torch.save(model.state_dict(), save_last_path)

        save_history_path = save_dir / "history.json"
        with open(save_history_path, "w") as f:
            json.dump(history, f, indent=1, default=lambda x: x.tolist())

        logger.info(f"Last model and history saved to {save_checkpoint_dir}")
    
    model.load_state_dict(best_model)
    
    logger.info("Training complete")
    
    return model, history, class_names