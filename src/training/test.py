"""
Testing pipeline for image classification models.
"""
import logging
from pathlib import Path
import json

import torch
import torch.nn as nn

from utils.data import get_dataloader
from engine import eval_one_epoch
from models import get_model
from utils.metrics import compute_class_metrics, compute_advanced_metrics

logger = logging.getLogger(__name__)


def run_test(
        cfg: dict,
        model: nn.Module | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> dict:
    """
    Evaluates a model on the test dataset and returns the computed metrics

    Parameters
    ----------
    cfg : dict
        Configuration dictionary. Expected top-level keys:

        - ``seed`` : int, optional
            Global random seed passed to the dataloader.
        - ``save_dir`` : str or Path, optional
            Root directory to save ``test_metrics``.
        - ``model`` : dict
            Passed directly to ``get_model``. Must contain at least ``name``
            and ``num_classes``. Only accessed when ``checkpoint_path`` is
            provided instead of ``model``.
        - ``data`` : dict
            Passed directly to ``get_dataloader``. Must contain dataset paths,
            normalization stats, and loader settings.
        - ``eval`` : dict
            Contains ``advanced_metrics``, if is set to ``True`` macro-averaged
            and per-class advanced metrics are computed.
    model : nn.Module, optional
        Model to evaluate. If None, model will be loaded from ``checkpoint_path``.
    checkpoint_path : str or Path, optional
        Path to a checkpoint file containing 'model_state_dict'. Used only
        if model is None.

    Returns
    -------
    dict
        Dictionary containing test metrics with the following keys:

        - ``loss`` : float
            Average loss across all samples.
        - ``accuracy`` : float 
            Accuracy across all samples.
        - ``targets`` : torch.Tensor
            Concatenated true labels for all batches.
        - ``preds`` : torch.Tensor
            Concatenated predicted class indices for all batches.

        When ``advanced_metrics`` is enabled, the following keys are also
        present:

        - ``macro_precision`` : float
            Macro-averaged precision across all classes.
        - ``macro_recall`` : float
            Macro-averaged recall across all classes.
        - ``macro_f1`` : float
            Macro-averaged F1-score across all classes.
        - ``confusion_matrix`` : np.ndarray
            Confusion matrix of shape ``(C, C)``, where ``C`` is the number
            of classes.
        - ``class_precision`` : list of float
            Per-class precision score.
        - ``class_recall`` : list of float
            Per-class recall score.
        - ``class_f1`` : list of float
            Per-class f1 score.

    Raises
    ------
    ValueError
        If both ``model`` and ``checkpoint_path`` are ``None``.
    """
    seed = cfg.get("seed", None)
    test_dataloader = get_dataloader(cfg["data"], seed=seed)["test_loader"]
    class_names = test_dataloader.dataset.classes
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model is None:
        if checkpoint_path is not None:
            checkpoint = torch.load(
                checkpoint_path,
                weights_only=True,
                map_location=device,
            )
            model = get_model(cfg["model"])
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise ValueError(
                "No model found. Either model or checkpoint_path must be provided"
            )
    model = model.to(device)

    model.eval()
    metrics = eval_one_epoch(
        dataloader=test_dataloader,
        model=model,
        criterion=criterion,
        device=device,
        )
    
    if cfg["test"].get("advanced_metrics", False):
        logger.info("Advanced metrics are active")
        targets = metrics["targets"]
        preds = metrics["preds"]

        macro_metrics = compute_advanced_metrics(targets=targets, preds=preds)
        class_metrics = compute_class_metrics(targets=targets, preds=preds)
    
        test_metrics = metrics | macro_metrics | class_metrics
    
    else:
        test_metrics = metrics

    save_dir = cfg.get("save_dir", None)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "test_metrics.json"
        with open(save_path, "w") as f:
            json.dump(test_metrics, f, indent=1, default=lambda x: x.tolist())
        logger.info(f"Test metrics saved to {save_dir}")

    return test_metrics, class_names