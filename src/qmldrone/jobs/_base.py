from ..metrics._base import train as common_train, test as common_test
from ..io._input import dataloader
import torchvision.datasets as ds
import torch
import os


def train(
    model_builder,
    conf,
    model_dir: str,
    train_data_dir: str,
):
    os.system(f"mkdir -p {model_dir}")
    for snr in conf["snr"]:
        cur_model_path = f"{model_dir}/{conf['model_name']}-{snr}.pt"
        trainset_root = f"{train_data_dir}/{conf['f_s']}fs/{snr}SNR"
        print(trainset_root)
        trainds = ds.DatasetFolder(trainset_root, dataloader, extensions=("npy",))
        trainLoader = torch.utils.data.DataLoader(
            trainds, conf["batch_size"], shuffle=True, num_workers=2
        )

        print(f"SNR: {snr} dB")
        net = model_builder(conf)
        common_train(conf, net, cur_model_path, trainLoader)


def test(
    model_builder,
    conf,
    classes,
    model_dir: str,
    plot_dir: str,
    test_data_dir: str,
):
    os.system(f"mkdir -p {plot_dir}")
    for snr in conf["snr"]:
        print(test_data_dir)
        cur_model_path = f"{model_dir}/{conf['model_name']}-{snr}.pt"
        testset_root = f"{test_data_dir}/{conf['f_s']}fs/{snr}SNR"
        print(testset_root)
        testds = ds.DatasetFolder(testset_root, dataloader, extensions=("npy",))
        testLoader = torch.utils.data.DataLoader(
            testds, conf["batch_size"], shuffle=True, num_workers=2
        )

        print(f"SNR: {snr} dB")

        net = model_builder(conf)
        common_test(
            conf, net, snr, cur_model_path, testLoader, classes, plot_dir=plot_dir
        )
