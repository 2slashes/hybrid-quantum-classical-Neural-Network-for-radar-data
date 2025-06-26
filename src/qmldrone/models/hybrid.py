import numpy as np
import torch
import torch.nn.functional as F
import pennylane as qml
from .classic import ClassicalRadarClassifier
from ._base import create_base
from ..io._input import get_num_qlayers_from_yaml


class HybridRadarClassifier(ClassicalRadarClassifier):
    def __init__(self, num_outputs, circuit, weight_shapes, n_qlayers):
        super(HybridRadarClassifier, self).__init__(num_outputs)

        # initialize the required number of qlayers
        self.qlayers = []
        for _ in range(n_qlayers):
            self.qlayers.append(qml.qnn.TorchLayer(circuit, weight_shapes))

    def forward(self, x):
        # i/p shape - (batch_size, Channel_in, Height_in, Width_in) - (2, 16, 251)
        x = self.pool1(self.relu(self.IN1(self.conv1(x))))
        x = self.drop(x)
        x = self.pool2(self.relu(self.IN2(self.conv2(x))))
        x = self.drop(x)
        # flatten
        x = x.view(-1, 32 * 12 * 21)  # x = torch.flatten(x, start_dim=1
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # qlayers connected parallely
        # Make sure the values are normalised to lie in the range [0,pi]
        # since the qnodes are using angle embedding
        x = F.normalize(x) * np.pi

        x_split = torch.split(x, 5, dim=1)
        post_qlayer = []
        for chunk, qlayer in zip(x_split, self.qlayers):
            post_qlayer.append(qlayer(chunk))

        x = torch.cat(post_qlayer, axis=1)

        # if we want qlayers connected serially?
        # x = self.qlayer1(x)
        # x = self.qlayer2(x)
        # x = self.qlayer3(x)
        # x = self.qlayer4(x)

        x = self.fc3(x)
        return x


def create(yaml_path: str, qlayer):
    circuit = qlayer.circuit
    weight_shapes = qlayer.weight_shapes
    num_qlayers = get_num_qlayers_from_yaml(yaml_path)
    create_base(
        yaml_path,
        lambda n: HybridRadarClassifier(n, circuit, weight_shapes, num_qlayers),
    )
