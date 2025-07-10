import numpy as np
import torch
import torch.nn.functional as F
import pennylane as qml
from .classic import RadarClassifier
from ._base import create_base
from ..io._input import get_num_qlayers_from_yaml
from .quantum import OriginalCircuit, QuantumLayer


class HybridRadarClassifier(RadarClassifier):
    def __init__(self, num_outputs, circuit, weight_shapes, n_qlayers):
        super(HybridRadarClassifier, self).__init__(num_outputs)

        self.qlayers = []
        for _ in range(n_qlayers):
            self.qlayers.append(qml.qnn.TorchLayer(circuit, weight_shapes))

    def forward(self, x):
        x = self.pool1(self.relu(self.IN1(self.conv1(x))))
        x = self.drop(x)
        x = self.pool2(self.relu(self.IN2(self.conv2(x))))
        x = self.drop(x)

        x = x.view(-1, 32 * 12 * 21)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        x = F.normalize(x) * np.pi

        x_split = torch.split(x, 5, dim=1)
        post_qlayer = []
        for chunk, qlayer in zip(x_split, self.qlayers):
            post_qlayer.append(qlayer(chunk))

        x = torch.cat(post_qlayer, axis=1)

        x = self.fc3(x)
        return x


def create(yaml_path: str, qlayer: QuantumLayer = OriginalCircuit()) -> None:
    circuit = qlayer.circuit
    weight_shapes = qlayer.weight_shapes
    num_qlayers = get_num_qlayers_from_yaml(yaml_path)
    create_base(
        yaml_path,
        lambda n: HybridRadarClassifier(n, circuit, weight_shapes, num_qlayers),
    )
