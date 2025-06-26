from ..metrics._base import test_metrics
from ..io._input import dataloader
import torchvision.datasets as ds
import torch
import os
import torch.nn as nn


def train(
    model_builder,
    conf,
    num_outputs,
    device,
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
        net = model_builder(num_outputs)

        net = net.to(device)
        optim = torch.optim.AdamW(net.parameters(), lr=conf["learning_rate"])
        loss_fn = nn.CrossEntropyLoss().to(device)
        for x in range(conf["epochs"]):
            net.train()

            for i, data in enumerate(trainLoader):
                # plot_spectrogram(torch.squeeze(data[0]))
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optim.zero_grad()

                outputs = net(inputs.float())
                loss = loss_fn(outputs, labels)
                loss.backward()
                optim.step()
                loss_val = loss.item()

            print("Train Epoch: {} Loss: {:.6f}".format(x, loss_val))
            if x > conf["min_epochs"] and loss_val < conf["loss_threshold"]:
                break

        if conf["save_model"] is True:
            print(f"Saving model state to {cur_model_path}")
            torch.save(net.state_dict(), cur_model_path)


def test(
    model_builder,
    conf,
    num_outputs,
    classes,
    device,
    model_dir: str,
    plot_dir: str,
    test_data_dir: str,
):
    os.system(f"mkdir -p {plot_dir}")
    metrics_conf = dict(
        (key, conf[key])
        for key in (
            "model_type",
            "plot_confusion",
        )
    )
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

        net = model_builder(num_outputs)
        test_metrics(
            metrics_conf,
            net,
            snr,
            cur_model_path,
            testLoader,
            classes,
            device,
            plot_dir=plot_dir,
        )
