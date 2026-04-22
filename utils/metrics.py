"""
Metrics computation for multiclass classification tasks.
"""
import logging
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


logger = logging.getLogger(__name__)

def compute_metrics(targets: torch.Tensor, preds: torch.Tensor) -> dict:
    """
    Compute accuracy, precision, recall, F1-score, and confusion matrix.

    Parameters
    ----------
    targets : torch.Tensor
        True class labels, shape ``(N,)``.
    preds : torch.Tensor
        Predicted class indices, shape ``(N,)``.

    Returns
    -------
    dict
        ``accuracy`` (float): fraction of correctly classified samples.
        ``precision`` (float): macro-averaged precision across all classes.
        ``recall`` (float): macro-averaged recall across all classes.
        ``f1_score`` (float): macro-averaged F1-score across all classes.
        ``confusion_matrix`` (np.ndarray): confusion matrix of shape ``(C, C)``,
        where ``C`` is the number of classes.

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
    >>> compute_metrics(targets, preds)
    {'accuracy': 0.6667, 'precision': 0.6667, 'recall': 0.5, 'f1_score': 0.5556,
     'confusion_matrix': array([[1, 0, 0],
                                [0, 0, 0],
                                [0, 1, 1]])}
    """
    # Shape check
    if targets.shape != preds.shape:
        msg = f"Predictions(preds) and Targets(targets) shapes do not match: {targets.shape} vs {preds.shape}"
        logger.error(msg)
        raise ValueError(msg)

    #Convert tensors to NumPy arrays for sklearn
    targets = targets.cpu().numpy()
    preds = preds.cpu().numpy()

    accuracy = round(float((preds==targets).sum() / len(targets)), 4)
    precision = round(float(precision_score(targets, preds, average="macro", zero_division=0)), 4)
    recall = round(float(recall_score(targets, preds, average="macro", zero_division=0)), 4)
    f1 = round(float(f1_score(targets, preds, average="macro", zero_division=0)), 4)
    conf_matrix = confusion_matrix(targets, preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix
    }


def main():
    # Test compute_metrics
    preds = torch.tensor([0,1,2])
    targets = torch.tensor([0,2,2])

    metrics = compute_metrics(targets, preds)
    
    for k,v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()