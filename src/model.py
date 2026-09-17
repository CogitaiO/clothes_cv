import torch.nn as nn
from torchvision import models

class ClothesModel(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.resnet18(weights="DEFAULT")

        for param in self.backbone.parameters():
            param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)