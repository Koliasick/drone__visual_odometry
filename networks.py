import torch
import torchvision.models as models
from torch import nn
from torchvision.models import ResNet18_Weights


class DualResnetModel(nn.Module):
    def __init__(self):
        super(DualResnetModel, self).__init__()
        # Convolutional layers

        # Convolutional big
        self.resnet1 = models.resnet18(pretrained=True)

        # Freeze the convolutional layers
        #for param in self.resnet1.parameters():
        #    param.requires_grad = False

        in_ft = self.resnet1.fc.in_features
        self.resnet1.fc = torch.nn.Identity()

        # Convolutional small
        self.resnet2 = models.resnet18(pretrained=True)

        # Freeze the convolutional layers
        #for param in self.resnet2.parameters():
        #    param.requires_grad = False

        self.resnet2.fc = torch.nn.Identity()

        # Fully connected layers
        self.lay1 = nn.Linear(in_ft * 2, 1000)
        self.relu = nn.ReLU()
        self.lay2 = nn.Linear(1000, 3)

    def forward(self, x):
        drone_image, satellite_image = torch.split(x, [3, 3], dim=1)
        bg = self.resnet1(satellite_image)
        sm = self.resnet2(drone_image)
        fc_input = torch.cat((sm, bg), dim=1)
        res = self.lay2(self.relu(self.lay1(fc_input)))
        return res
