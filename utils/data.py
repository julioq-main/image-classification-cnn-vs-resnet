"""
utils/data.py

API to get dataloader
"""


#Importing libraries
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

#Setting transform functions
def build_transforms(mean: list, std: list):

    train_transform = transforms.Compose([transforms.Resize(256), 
                        transforms.CenterCrop(224), 
                        transforms.RandomHorizontalFlip(0.5), 
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])

    test_val_transform = transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
    return train_transform, test_val_transform


#Get dataloader
def get_dataloader(cfg: dict) -> dict[DataLoader]:
    #Defining parameters using yaml config
    mean = cfg["mean"]
    std = cfg["std"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]

    #Building dataset transformations
    train_transform, test_val_transform = build_transforms(mean,std)

    #Setting datasets
    train_dataset= datasets.ImageFolder(root=cfg['train_dir'], transform=train_transform)
    test_dataset = datasets.ImageFolder(root=cfg['test_dir'], transform=test_val_transform)
    val_dataset = datasets.ImageFolder(root=cfg['val_dir'], transform=test_val_transform)

    #Setting dataloaders (in GPU if possible)
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return {"train_loader":train_loader, "test_loader":test_loader, "val_loader":val_loader}