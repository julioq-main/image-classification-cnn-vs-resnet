"""
models/architectures.py

"""


#Import ResNet18 model
from torchvision.models import resnet18, ResNet18_Weights 
import torch.nn as nn


class MyResNet18(nn.Module):
#Defining the model and tuning it to match our task
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        
        #Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
