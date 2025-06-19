import torch
import torch.nn as nn
import torchvision.datasets as ds
import os
from .common import (
    train as common_train,
    test as common_test,
    dataloader,
)
from .config import get_conf, get_paths


class ClassicalRadarClassifier(nn.Module):
    def __init__(self, conf):
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
        self.fc3 = nn.Linear(20, 5)  # o/p shape should be (batch_size,5,1,1)
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


# ----------------------TRAINING CODE-----------------------


def train(conf):
    paths = get_paths()

    os.system(f"mkdir -p {paths['model_dir']}")
    for snr in conf["snr"]:
        cur_model_path = f"{paths['model_dir']}/{conf['model_name']}-{snr}.pt"
        # assumes train data is split by SNR
        trainset_root = f"{paths['train_data_dir']}/{conf['f_s']}fs/{snr}SNR"
        trainds = ds.DatasetFolder(trainset_root, dataloader, extensions=("npy",))
        trainLoader = torch.utils.data.DataLoader(
            trainds, conf["batch_size"], shuffle=True, num_workers=2
        )

        print(f"SNR: {snr} dB")
        net = ClassicalRadarClassifier(conf)
        common_train(conf, net, cur_model_path, trainLoader)


# ---------------------EVALUATION------------------------------


def test(conf):
    paths = get_paths()

    os.system(f"mkdir -p {paths['plot_dir']}")
    for snr in conf["snr"]:
        cur_model_path = f"{paths['model_dir']}/{conf['model_name']}-{snr}.pt"
        # assumes the test data is split by SNR
        testset_root = f"{paths['test_data_dir']}/{conf['f_s']}fs/{snr}SNR"
        testds = ds.DatasetFolder(testset_root, dataloader, extensions=("npy",))
        testLoader = torch.utils.data.DataLoader(
            testds, conf["batch_size"], shuffle=True, num_workers=2
        )

        print(f"SNR: {snr} dB")

        net = ClassicalRadarClassifier(conf)
        common_test(
            conf, net, snr, cur_model_path, testLoader, plot_dir=paths["plot_dir"]
        )

def create():
    conf = get_conf()
    train(conf)
    test(conf)