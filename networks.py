import torch
import torchvision.models as models
from torch import nn


class ResNetModified(torch.nn.Module):
    def __init__(self):
        super(ResNetModified, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = torch.nn.Conv2d(6, 64, kernel_size=7, stride=3, padding=3, bias=False)
        self.resnet.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(1000, 3)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DroneSatelliteModelAttempt2(nn.Module):
    def __init__(self):
        super(DroneSatelliteModelAttempt2, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 37 * 37, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn6 = nn.BatchNorm1d(64)
        # Output layers
        self.classifier = nn.Linear(64, 1)
        self.regressor = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool1(self.bn1(nn.functional.relu(self.conv1(x))))
        x = self.pool2(self.bn2(nn.functional.relu(self.conv2(x))))
        x = self.pool3(self.bn3(nn.functional.relu(self.conv3(x))))
        x = self.pool4(self.bn4(nn.functional.relu(self.conv4(x))))
        x = x.view(-1, 128 * 37 * 37)
        x = self.bn5(nn.functional.relu(self.fc1(x)))
        x = self.bn6(nn.functional.relu(self.fc2(x)))
        c = torch.sigmoid(self.classifier(x))
        coordinates = self.regressor(x)
        return torch.cat((c, coordinates), dim=1)