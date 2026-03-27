"""
utils/metrics.py

Function to compute advanced metrics for classification tasks

Metrics included:
 - Accuracy
 - Precision
 - Recall
 - F1 Score
 - Confusion Matrix

Metrics are computed for each epoch
"""

import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def compute_metrics(targets: torch.Tensor, preds: torch.Tensor) -> dict:
    """
    Computes all metrics

    Args:
        targets (torch.Tensor): The true labels for the epoch, shape (N,)
        preds (torch.Tensor): The predictions for the epoch, shape (N,)

    Returns:
        dict: {
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1": float
            "confusion_matrix": np.ndarray
        }
    
    Raises:
        ValueError: Predictions and Targets shapes do not match

    Notes:
        - Uses zero_division=0 to handle classes with no predictions.
        - Macro average gives equal weight to each class.


    Example:
        >>> preds = torch.tensor([0,1,2])
        >>> targets = torch.tensor([0,2,2])
        >>> compute_metrics(targets,preds)
        {'accuracy': 0.6667, 'precision': 0.6667, 'recall': 0.5, 'f1_score': 0.5556, 
         'confusion_matrix': array([[1, 0, 0],
                                    [0, 0, 0],
                                    [0, 1, 1]])}

    """

    #Convert tensors to NumPy arrays for sklearn
    targets = targets.cpu().numpy()
    preds = preds.cpu().numpy()
    
    # Shape check
    if targets.shape != preds.shape:
        raise ValueError(f"Predictions(preds) and Targets(targets) shapes do not match: {targets.shape} vs {preds.shape}")

    # Compute metrics
    accuracy = (preds==targets).sum()/len(targets)
    precision = precision_score(targets, preds,average="macro", zero_division=0)
    recall = recall_score(targets, preds, average="macro", zero_division=0)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    conf_matrix = confusion_matrix(targets, preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix}


def main():
    # Test compute_metrics
    preds = torch.tensor([0,1,2])
    targets = torch.tensor([0,2,2])

    metrics = compute_metrics(targets, preds)
    
    for k,v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()