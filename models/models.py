"""
TODO add more models
TODO doc


"""


#Import ResNet18 model
from torchvision.models import resnet18, ResNet18_Weights 
from torchvision import models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class MyResNet18(nn.Module):
#Defining the model and tuning it to match our problem
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)                            #Changing the last layer to output 25 classes not 1000 as the standard does

    def forward(self, x):
        return self.model(x)


class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        
        #Convolutional layers
        conv1 = nn.Conv2d(3,32,kernel_size=3,padding=1)
