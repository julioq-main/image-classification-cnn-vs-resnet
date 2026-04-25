"""
Training and evaluation loops for classification models in PyTorch.
"""
import torch
from tqdm import tqdm


def train_one_epoch(
        dataloader: torch.utils.data.DataLoader, 
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> dict:
    """
    Run one epoch of training and return loss and accuracy.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader for the training set.
    model : torch.nn.Module
        Model to train.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer for model parameters.
    device : torch.device
        Device to perform computations on.

    Returns
    -------
    dict
        Loss and Accuracy from training one epoch

        - ``loss`` : float 
            Average loss across all samples.
        - ``accuracy`` : float
            Accuracy across all samples.
    """
    model.train()

    total_loss, total_correct = 0.0,0
    size = len(dataloader.dataset)
    
    for (images, targets) in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        preds = model(images)
        loss = criterion(preds,targets)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)  # Handles last batch if smaller
        total_loss += loss.item()*batch_size  # Take into account batch size
        total_correct += (preds.argmax(1)==targets).sum().item()  # Comparing real target and prediction

    avg_loss = total_loss/size
    accuracy = total_correct/size

    return {
        "loss":avg_loss,
        "accuracy":accuracy
    }

@torch.no_grad()
def eval_one_epoch(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        device: torch.device
    ) -> dict:
    """
    Run one epoch of evaluation and return loss, accuracy, and raw predictions.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader for the validation or test set.
    model : torch.nn.Module
        Model to evaluate.
    criterion : torch.nn.Module
        Loss function.
    device : torch.device
        Device to perform computations on.

    Returns
    -------
    dict
        Basic metrics and raw predictions. Keys:

        - ``loss`` : float
            Average loss across all samples.
        - ``accuracy`` : float 
            Accuracy across all samples.
        - ``targets`` : torch.Tensor
            Concatenated true labels for all batches.
        - ``preds`` : torch.Tensor
            Concatenated predicted class indices for all batches.

    Notes
    -----
    Decorated with ``@torch.no_grad()``, so no gradients are computed or
    stored during the forward pass.

    ``targets`` and ``preds`` are moved to CPU and concatenated across all
    batches, allowing computation of additional metrics such as precision,
    recall, F1-score, and confusion matrix after the call.
    """    
    model.eval()

    size = len(dataloader.dataset)
    total_loss, total_correct = 0.0, 0
    all_preds, all_targets = [], []
    
    #Iterating for each batch
    for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        preds = model(images)
        pred_classes = preds.argmax(1)
        loss = criterion(preds, targets)

        all_preds.append(pred_classes.cpu())
        all_targets.append(targets.cpu())

        batch_size = images.size(0)  # Handles last batch if smaller
        total_loss += loss.item()*batch_size  # Take into account batch size
        total_correct += (pred_classes==targets).sum().item()  #Comparing real target and the prediction

    avg_loss = total_loss/size
    accuracy = total_correct/size
    
    output_targets = torch.cat(all_targets)
    output_preds = torch.cat(all_preds)

    return {
        "loss":avg_loss,
        "accuracy":accuracy,
        "targets": output_targets,
        "preds": output_preds
    }
