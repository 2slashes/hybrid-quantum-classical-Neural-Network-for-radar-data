from qmldrone.common import common_train, common_test, get_paths, dataloader
import torchvision.datasets as ds
import torch
import os


def train(conf, model_builder, model_dir: str="../original/Radartwo_sided_models", train_data_dir: str="../original/Radartwo_sided_train_data"):

    os.system(f"mkdir -p {model_dir}")
    for snr in conf["snr"]:
        cur_model_path = f"{model_dir}/{conf['model_name']}-{snr}.pt"
        trainset_root = f"{paths['train_data_dir']}/{conf['f_s']}fs/{snr}SNR"
        trainds = ds.DatasetFolder(trainset_root, dataloader, extensions=("npy",))
        trainLoader = torch.utils.data.DataLoader(
            trainds, conf["batch_size"], shuffle=True, num_workers=2
        )

        print(f"SNR: {snr} dB")
        net = model_builder(conf)
        common_train(conf, net, cur_model_path, trainLoader)

def test(conf, model_builder):
    paths = get_paths()

    os.system(f"mkdir -p {paths['plot_dir']}")
    for snr in conf["snr"]:
        cur_model_path = f"{paths['model_dir']}/{conf['model_name']}-{snr}.pt"
        testset_root = f"{paths['test_data_dir']}/{conf['f_s']}fs/{snr}SNR"
        testds = ds.DatasetFolder(testset_root, dataloader, extensions=("npy",))
        testLoader = torch.utils.data.DataLoader(
            testds, conf["batch_size"], shuffle=True, num_workers=2
        )

        print(f"SNR: {snr} dB")

        net = model_builder(conf)
        common_test(
            conf, net, snr, cur_model_path, testLoader, plot_dir=paths["plot_dir"]
        )