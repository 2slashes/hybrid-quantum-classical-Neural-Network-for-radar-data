import torch
import torch.nn as nn
from ._base import create_base


class ClassicalRadarClassifier(nn.Module):
    def __init__(self, conf: dict):
        super(ClassicalRadarClassifier, self).__init__()
        self.outputs = conf["num_outputs"]
        # i/p shape - (batch_size, Channel_in, Height_in, Width_in) - (2, 16, 251)
        self.conv1 = nn.Conv2d(2, 16, (3, 3), padding=1)  # o/p shape - (16, 16, 251)
        self.IN1 = nn.InstanceNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2, padding=1)  # o/p shape (16, 8, 126)
        self.conv2 = nn.Conv2d(16, 32, (5, 5), padding=2)  # o/p shape (32, 8, 126)
        self.IN2 = nn.InstanceNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # o/p shape (32, 4, 63)
        # fully connected layers
        self.fc1 = nn.Linear(32 * 4 * 63, 120)
        self.fc2 = nn.Linear(120, 20)
        self.fc3 = nn.Linear(20, self.outputs)  # o/p shape should be (batch_size,5,1,1)
        self.drop = nn.Dropout2d(p=0.5)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # i/p shape - (batch_size, Channel_in, Height_in, Width_in) - (2, 16, 251)
        x = self.pool1(self.relu(self.IN1(self.conv1(x))))
        x = self.drop(x)
        x = self.pool2(self.relu(self.IN2(self.conv2(x))))
        x = self.drop(x)
        # flatten
        x = torch.flatten(x, start_dim=1)
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create(conf_path: str):
    create_base(conf_path, lambda c: ClassicalRadarClassifier(c))
