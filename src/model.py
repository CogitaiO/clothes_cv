import torch.nn as nn
from torchvision import models

class ClothesModel(nn.Module):

     def __init__(self, num_classes):
         super().__init__()

         self.model = models.resnet18(weights="DEFAULT")

         for param in self.model.parameters():
             param.requires_grad = False
             
         in_features = self.model.fc.in_features
         self.model.fc = nn.Linear(in_features, num_classes)

     def forward(self, x):
         return self.model(x)