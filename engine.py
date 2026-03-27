"""
engine/engine.py

Provides training and evaluation loops for classification models in PyTorch.

Functions:
- train_one_epoch: Runs one epoch of training and computes loss and accuracy.
- eval_one_epoch: Evaluates the model for one epoch, returning loss, accuracy, labels and predictions.

Notes:
- Loss is averaged across all samples.
- Accuracy is computed per sample.
- For evaluation, predictions and targets are concatenated across batches to allow computation of
  additional metrics (precision, recall, F1, confusion matrix).
- tqdm is used for progress visualization.
"""

import torch
from tqdm import tqdm

#train loop
def train_one_epoch(dataloader: torch.utils.data.DataLoader, 
                    model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> dict:
    """
    Performs one epoch of training.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        model (torch.nn.Module): The model to train.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device (torch.device): Device to perform computations on (CPU/GPU).

    Returns:
        dict: {
            "loss": float,       # Average loss across all samples
            "accuracy": float    # Accuracy across all samples
        }

    Behavior:
        - Model is set to train mode.
        - Gradients are computed and backpropagated for each batch.
        - Loss and correct predictions are accumulated per batch.
        - Handles varying batch sizes (e.g., last batch may be smaller).
    """

    model.train()  # Enable training mode

    total_loss, total_correct = 0.0,0
    size = len(dataloader.dataset)  # Total number of samples
    
    #Iterating for each batch
    for (images, targets) in tqdm(dataloader, desc="Training", leave=False):
        #Getting images, targets, the target prediction and the loss
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward Pass
        preds = model(images)
        loss = criterion(preds,targets)

        # Backward Pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Accumulate Batch Metrics
        batch_size = images.size(0)  # Handles last batch if smaller
        total_loss += loss.item()*batch_size  # Take into account batch size
        total_correct += (preds.argmax(1)==targets).sum().item()  # Comparing real target and prediction

    avg_loss = total_loss/size
    accuracy = total_correct/size

    return {"loss":avg_loss, "accuracy":accuracy}

@torch.no_grad()  #Don't compute gradients
def eval_one_epoch(dataloader: torch.utils.data.DataLoader,
                   model: torch.nn.Module,
                   criterion: torch.nn.Module,
                   device: torch.device) -> dict:
    """
    Performs one epoch of evaluation (validation or testing).

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        model (torch.nn.Module): The model to evaluate.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to perform computations on (CPU/GPU).

    Returns:
        dict: {
            "loss": float,           # Average loss across all samples
            "accuracy": float,       # Accuracy across all samples
            "targets": torch.Tensor, # Concatenated true labels for all batches
            "preds": torch.Tensor    # Concatenated predicted labels for all batches
        }

    Behavior:
        - Model is set to evaluation mode.
        - No gradients are computed (saves memory and computation).
        - Predictions and targets are stored per batch and concatenated to allow computation
          of additional metrics like precision, recall, F1-score, and confusion matrix.
        - Handles varying batch sizes.
    """
    
    model.eval()  # Enable evaluation mode

    size = len(dataloader.dataset)
    total_loss, total_correct = 0.0, 0
    total_preds, total_targets = [], []
    
    #Iterating for each batch
    for images, targets in tqdm(dataloader,desc="Evaluating", leave=False):
        #Getting images, targets, target prediction (in GPU if possible)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward Pass
        preds = model(images)
        loss = criterion(preds, targets)

        # Store predictions and targets for metrics computation
        total_preds.append(preds.argmax(1))
        total_targets.append(targets)

        # Accumulate batch metrics
        batch_size = images.size(0)  # Handles last batch if smaller
        total_loss += loss.item()*batch_size  # Take into account batch size
        total_correct += (preds.argmax(1)==targets).sum().item()  #Comparing real target and the prediction

    avg_loss = total_loss/size
    accuracy = total_correct/size
    
    # Concatenate all batches for global metrics
    output_targets = torch.cat(total_targets)
    output_preds = torch.cat(total_preds)

    return {"loss":avg_loss, "accuracy":accuracy, "targets": output_targets, "preds": output_preds}
