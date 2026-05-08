"""
Metrics computation for multiclass classification tasks.
"""
import logging
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)


def compute_advanced_metrics(targets: torch.Tensor, preds: torch.Tensor) -> dict:
    """
    Compute macro-averaged accuracy, precision, recall and F1, and confusion matrix.

    Parameters
    ----------
    targets : torch.Tensor
        True class labels, shape ``(N,)``.
    preds : torch.Tensor
        Predicted class indices, shape ``(N,)``.

    Returns
    -------
    dict
        Evaluation metrics. Keys:
        
        - ``macro_precision`` : float
            Macro-averaged precision across all classes.
        - ``macro_recall`` : float
            Macro-averaged recall across all classes.
        - ``macro_f1`` : float
            Macro-averaged F1-score across all classes.
        - ``confusion_matrix`` : np.ndarray
            Confusion matrix of shape ``(C, C)``, where ``C`` is the number
            of classes.

    Raises
    ------
    ValueError
        If ``targets`` and ``preds`` have different shapes.

    Notes
    -----
    Macro averaging gives equal weight to each class regardless of support.
    Classes with no predicted samples contribute 0 to the average
    (``zero_division=0``).

    All float metrics are rounded to 4 decimal places.

    Examples
    --------
    >>> preds = torch.tensor([0, 1, 2])
    >>> targets = torch.tensor([0, 2, 2])
    >>> compute_advanced_metrics(targets, preds)
    {'accuracy': 0.6667, 'precision': 0.6667, 'recall': 0.5, 'f1_score': 0.5556,
     'confusion_matrix': array([[1, 0, 0],
                                [0, 0, 0],
                                [0, 1, 1]])}
    """
    # Shape check
    if targets.shape != preds.shape:
        raise ValueError(
            f"Predictions(preds) and Targets(targets) shapes do not match: {targets.shape} vs {preds.shape}"
        )

    #Convert tensors to NumPy arrays for sklearn
    targets = targets.cpu().numpy()
    preds = preds.cpu().numpy()

    macro_precision = round(float(precision_score(targets, preds, average="macro", zero_division=0)), 4)
    macro_recall = round(float(recall_score(targets, preds, average="macro", zero_division=0)), 4)
    macro_f1 = round(float(f1_score(targets, preds, average="macro", zero_division=0)), 4)
    conf_matrix = confusion_matrix(targets, preds)
    
    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": conf_matrix
    }


def compute_class_metrics(targets: torch.Tensor, preds: torch.Tensor) -> dict[list[float]]:
    """
    Compute per-class precision recall and F1.

    Parameters
    ----------
    targets : torch.Tensor
        True class labels, shape ``(N,)``.
    preds : torch.Tensor
        Predicted class indices, shape ``(N,)``.

    Returns
    -------
    dict
        Per-class metrics. Keys:

        - ``class_precision`` : list of float
            Per-class precision score.
        - ``class_recall`` : list of float
            Per-class recall score.
        - ``class_f1`` : list of float
            Per-class f1 score.
    
    Raises
    ------
    ValueError
        If ``targets`` and ``preds`` have different shapes.

    Notes
    -----
    Per-class metrics are to be computed during test, not to be computed at
    each epoch of training. All float metrics are rounded to 4 decimal places.
    
    Examples
    --------
    >>> preds = torch.tensor([0, 1, 2])
    >>> targets = torch.tensor([0, 2, 2])
    >>> compute_class_metrics(targets, preds)
    {'class_precision': [1.0, 0.0, 1.0], 'class_recall': [1.0, 0.0, 0.5],
    'class_f1': [1.0, 0.0, 0.6667]}
    """
    # Shape check
    if targets.shape != preds.shape:
        msg = f"Predictions(preds) and Targets(targets) shapes do not match: {targets.shape} vs {preds.shape}"
        raise ValueError(msg)

    #Convert tensors to NumPy arrays for sklearn
    targets = targets.cpu().numpy()
    preds = preds.cpu().numpy()
    
    class_precision = np.around(precision_score(targets, preds, average=None, zero_division=0), 4).tolist()
    class_recall = np.around(recall_score(targets, preds, average=None, zero_division=0), 4).tolist()
    class_f1 = np.around(f1_score(targets, preds, average=None, zero_division=0), 4).tolist()
    
    return {
        "class_precision": class_precision,
        "class_recall": class_recall,
        "class_f1": class_f1,
    }

def main() -> None:
    # Test compute_metrics
    preds = torch.tensor([0,1,2])
    targets = torch.tensor([0,2,2])

    metrics = compute_advanced_metrics(targets, preds)
    class_metrics = compute_class_metrics(targets, preds)
    for k,v in metrics.items():
        print(f"{k}: {v}")
    
    for k,v in class_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()