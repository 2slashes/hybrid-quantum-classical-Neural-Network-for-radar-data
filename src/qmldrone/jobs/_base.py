from ..metrics._base import test_metrics
from ..io._input import dataloader
import torchvision.datasets as ds
import torch
import torch.nn as nn


def train(
    classifier,
    conf: dict[str, any],
    device,
    snr: int,
    train_data_dir: str,
    disable_min_epochs: bool = True,
) -> None:
    trainset_root = f"{train_data_dir}/{conf['f_s']}fs/{snr}SNR"
    trainds = ds.DatasetFolder(trainset_root, dataloader, extensions=("npy",))
    trainLoader = torch.utils.data.DataLoader(
        trainds, conf["batch_size"], shuffle=True, num_workers=2
    )

    print(f"SNR: {snr} dB")
    optim = torch.optim.AdamW(classifier.parameters(), lr=conf["learning_rate"])
    loss_fn = nn.CrossEntropyLoss().to(device)
    for x in range(conf["epochs"]):
        classifier.train()

        for i, data in enumerate(trainLoader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optim.zero_grad()

            outputs = classifier(inputs.float())
            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()
            loss_val = loss.item()

        print("Train Epoch: {} Loss: {:.6f}".format(x, loss_val))
        if (
            not disable_min_epochs
            and x > conf["min_epochs"]
            and loss_val < conf["loss_threshold"]
        ):
            break


def test(
    classifier,
    conf: dict[str, any],
    metrics_conf: dict[str, any],
    classes: list[str],
    device,
    snr: int,
    plot_dir: str,
    test_data_dir: str,
):
    testset_root = f"{test_data_dir}/{conf['f_s']}fs/{snr}SNR"
    testds = ds.DatasetFolder(testset_root, dataloader, extensions=("npy",))
    testLoader = torch.utils.data.DataLoader(
        testds, conf["batch_size"], shuffle=True, num_workers=2
    )

    print(f"SNR: {snr} dB")

    test_metrics(
        classifier,
        metrics_conf,
        snr,
        testLoader,
        classes,
        device,
        plot_dir=plot_dir,
    )
