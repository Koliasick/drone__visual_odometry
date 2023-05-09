import torch
import torchvision.models as models


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