import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as ds
import os
from .common import (
    train as common_train,
    test as common_test,
    dataloader,
)
from .config import get_conf, get_outputs, get_paths

# ----------------------------STUFF---------------------------------------


n_qlayers = 4


def set_qlayers(num_qlayers: int):
    # set the number of parallel qlayers
    global n_qlayers
    n_qlayers = num_qlayers


# ----------------------------MODEL DEFINITION----------------------------------

n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


# Define the QLayer
n_layers = 3
weight_shapes = {"weights": (n_layers, n_qubits)}


# This class is shared between the detection and classification models
# Detection model has 2 outputs, classification has 5 outputs, otherwise they are identical
# it might be possible that more outputs are desired for classification, maybe it should be an input
class HybridRadarClassifier(nn.Module):
    def __init__(self, conf):
        super(HybridRadarClassifier, self).__init__()
        self.outputs = get_outputs()
        # i/p shape - (batch_size, Channel_in, Height_in, Width_in) - (2, 16, 251)
        self.conv1 = nn.Conv2d(2, 16, (3, 3), padding=1)  # o/p shape - (16, 16, 251)
        self.IN1 = nn.InstanceNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2, padding=1)  # o/p shape (16, 8, 126)
        self.conv2 = nn.Conv2d(16, 32, (5, 5), padding=2)  # o/p shape (32, 8, 126)
        self.IN2 = nn.InstanceNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # o/p shape (32, 4, 63)

        # quantum layer
        # self.qlayer1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        # self.qlayer2 = qml.qnn.TorchLayer(qnode, weight_shapes)
        # self.qlayer3 = qml.qnn.TorchLayer(qnode, weight_shapes)
        # self.qlayer4 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayers = []
        global n_qlayers
        for i in range(n_qlayers):
            self.qlayers.append(qml.qnn.TorchLayer(qnode, weight_shapes))

        # fully connected layers
        self.fc1 = nn.Linear(32 * 12 * 21, 120)
        self.fc2 = nn.Linear(120, 5 * n_qlayers)
        self.fc3 = nn.Linear(
            5 * n_qlayers, self.outputs
        )  # o/p shape should be (batch_size,5,1,1)

        self.drop = nn.Dropout2d(p=0.5)
        self.relu = nn.LeakyReLU()

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
        global n_qlayers

        # x_1, x_2, x_3, x_4 = torch.split(x, 5, dim=1)
        x_split = torch.split(x, 5, dim=1)
        post_qlayer = []
        for chunk, qlayer in zip(x_split, self.qlayers):
            post_qlayer.append(qlayer(chunk))
        # print(f"input {x_1=}")
        # x_1 = self.qlayer1(x_1)
        # x_2 = self.qlayer2(x_2)
        # x_3 = self.qlayer3(x_3)
        # x_4 = self.qlayer4(x_4)
        # print(f"output {x_1=}")
        # x = torch.cat([x_1, x_2, x_3, x_4], axis=1)
        x = torch.cat(post_qlayer, axis=1)

        # if we want qlayers connected serially?
        # x = self.qlayer1(x)
        # x = self.qlayer2(x)
        # x = self.qlayer3(x)
        # x = self.qlayer4(x)

        x = self.fc3(x)
        return x


# ----------------------TRAINING CODE-----------------------


def train():
    conf = get_conf()
    paths = get_paths()

    os.system(f"mkdir -p {paths['model_path']}")
    for snr in conf["snr"]:
        cur_model_path = f"{paths['model_path']}/{conf['model_name']}-{snr}.pt"
        trainset_root = f"{paths['radar_path']}/{paths['data_dir']}/trainset/{conf['f_s']}fs/{snr}SNR"
        trainds = ds.DatasetFolder(trainset_root, dataloader, extensions=("npy",))
        trainLoader = torch.utils.data.DataLoader(
            trainds, conf["batch_size"], shuffle=True, num_workers=2
        )

        print(f"SNR: {snr} dB")
        net = HybridRadarClassifier(conf)
        common_train(conf, net, cur_model_path, trainLoader)


# ---------------------EVALUATION------------------------------


def test():
    conf = get_conf()
    paths = get_paths()

    os.system(f"mkdir -p {paths['plot_path']}")
    for snr in conf["snr"]:
        cur_model_path = f"{paths['model_path']}/{conf['model_name']}-{snr}.pt"

        testset_root = f"{paths['radar_path']}/{paths['data_dir']}/testset/{conf['f_s']}fs/{snr}SNR"
        testds = ds.DatasetFolder(testset_root, dataloader, extensions=("npy",))
        testLoader = torch.utils.data.DataLoader(
            testds, conf["batch_size"], shuffle=True, num_workers=2
        )

        print(f"SNR: {snr} dB")

        net = HybridRadarClassifier(conf)
        common_test(
            conf, net, snr, cur_model_path, testLoader, plot_dir=paths["plot_path"]
        )
