import torch
import torch.nn as nn
from ._base import create_base


class RadarClassifier(nn.Module):
    """A CNN-based classifier for radar data.

    This model is designed to process radar data with two channels and classify it into multiple output classes.
    
    Args:
        num_outputs (int): The number of output classes for classification.
    """
    def __init__(self, num_outputs):
        super(RadarClassifier, self).__init__()
        self.outputs = num_outputs
        self.conv1 = nn.Conv2d(2, 16, (3, 3), padding=1)
        self.IN1 = nn.InstanceNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, (5, 5), padding=2)
        self.IN2 = nn.InstanceNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 4 * 63, 120)
        self.fc2 = nn.Linear(120, 20)
        self.fc3 = nn.Linear(20, self.outputs)
        self.drop = nn.Dropout2d(p=0.5)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.IN1(self.conv1(x))))
        x = self.drop(x)
        x = self.pool2(self.relu(self.IN2(self.conv2(x))))
        x = self.drop(x)

        x = torch.flatten(x, start_dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create(yaml_path: str) -> None:
    create_base(yaml_path, lambda n: RadarClassifier(n))
