"""
API core train and test loop for each epoch
TODO
improve documentation
more metrics?
add logging when training after number of batches?
"""

import torch
from tqdm import tqdm

#train loop
def train_one_epoch(dataloader, model, criterion, optimizer, device):
    #Set model to train mode
    model.train()

    #Initialising variables to compute metrics (loss and accuracy)
    total_loss, total_correct = 0.0,0
    size_samples = len(dataloader.dataset)
    
    #Iterating the batches
    for (images, labels) in tqdm(dataloader, desc="Training"):
        #Getting images, labels, the label prediction and the loss
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        pred = model(images)
        loss = criterion(pred,labels)

        #Train one step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #Get some metrics (loss and accuracy)
        batch_size = images.size(0)                                                                 #Batch size can vary in the last batch, so better to take the the real size of each batch
        total_loss += loss.item()*batch_size                                                        #Multiply by batch_size to take into account batch size variation (probably not very big variation)
        total_correct += (pred.argmax(1)==labels).sum().item()                                      #Comparing real label and the predicted label

    avg_loss = total_loss/size_samples
    accuracy = total_correct/size_samples

    return {"loss":avg_loss, "accuracy":accuracy}

@torch.no_grad()                                                                                    #Don't compute gradients
def eval_one_epoch(dataloader, model, criterion, device):
    #Set modeltest mode
    model.eval()

    #Initialising variables to compute metrics (loss and accuracy)
    size_samples = len(dataloader.dataset)
    total_loss,total_correct = 0.0, 0
    
    for images, labels in tqdm(dataloader,desc="Evaluating"):
        #Getting images, labels, label prediction (in GPU if possible)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
            
        pred = model(images)
        loss = criterion(pred, labels)

        #Get some metrics (loss and accuracy)
        batch_size = images.size(0)                                                                 #Batch size can vary in the last batch, so better to take the the real size of each batch
        total_loss += loss.item()*batch_size                                                        #Multiply by batch_size to take into account batch size variation (probably not very big variation)
        total_correct += (pred.argmax(1)==labels).sum().item()                                      #Comparing real label and the predicted label

    avg_loss = total_loss/size_samples
    accuracy = total_correct/size_samples
    
    return {"loss":avg_loss, "accuracy":accuracy}
