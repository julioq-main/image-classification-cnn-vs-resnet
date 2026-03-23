"""
API to get dataloader
TODO Dataset sanity check(explore that)
TODO debug dataset (100 samples)?
TODO some data analysis? (PROBABLY IN ANOTHER SCRIPT data/scripts/analysis.py?)
TODO doc
TODO make augmentation possible through config
"""


#Importing libraries
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

#Setting transform functions
def build_transforms(mean: torch.Tensor, std: torch.Tensor):

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
    mean = cfg["data"]["mean"]
    std = cfg["data"]['std']
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    #Building dataset transformations
    train_transform, test_val_transform = build_transforms(mean,std)

    #Setting datasets
    train_dataset= datasets.ImageFolder(root=cfg["data"]['train_dir'], transform=train_transform)
    test_dataset = datasets.ImageFolder(root=cfg["data"]['test_dir'], transform=test_val_transform)
    val_dataset = datasets.ImageFolder(root=cfg["data"]['val_dir'], transform=test_val_transform)

    #Setting dataloaders (in GPU if possible)
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    return {"train_loader":train_loader, "test_loader":test_loader, "val_loader":val_loader}