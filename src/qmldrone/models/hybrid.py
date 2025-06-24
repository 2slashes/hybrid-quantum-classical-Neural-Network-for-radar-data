import numpy as np
import torch
import torch.nn.functional as F
import pennylane as qml
from .classic import ClassicalRadarClassifier
from ..io._input import get_conf


class HybridRadarClassifier(ClassicalRadarClassifier):
    def __init__(self, circuit, weight_shapes):
        super(HybridRadarClassifier, self).__init__()
        conf = get_conf()

        # initialize the required number of qlayers
        self.qlayers = []
        n_qlayers = conf["num_qlayers"]
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
