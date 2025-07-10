from ..metrics._base import test_metrics
from ..io._input import dataloader
import torchvision.datasets as ds
import torch
import torch.nn as nn


def train(
    classifier,
    model_config: dict[str, any],
    device,
    snr: int,
    train_data_dir: str,
    disable_min_epochs: bool = True,
) -> None:
    trainset_root = f"{train_data_dir}/{model_config['f_s']}fs/{snr}SNR"
    trainds = ds.DatasetFolder(trainset_root, dataloader, extensions=("npy",))
    trainLoader = torch.utils.data.DataLoader(
        trainds, model_config["batch_size"], shuffle=True, num_workers=2
    )

    print(f"SNR: {snr} dB")
    optim = torch.optim.AdamW(classifier.parameters(), lr=model_config["learning_rate"])
    loss_fn = nn.CrossEntropyLoss().to(device)
    for x in range(model_config["epochs"]):
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
            and x > model_config["min_epochs"]
            and loss_val < model_config["loss_threshold"]
        ):
            break


def test(
    classifier,
    model_config: dict[str, any],
    metrics_config: dict[str, any],
    drone_classes: list[str],
    device,
    snr: int,
    plot_dir: str,
    test_data_dir: str,
):
    testset_root = f"{test_data_dir}/{model_config['f_s']}fs/{snr}SNR"
    testds = ds.DatasetFolder(testset_root, dataloader, extensions=("npy",))
    testLoader = torch.utils.data.DataLoader(
        testds, model_config["batch_size"], shuffle=True, num_workers=2
    )

    print(f"SNR: {snr} dB")

    test_metrics(
        classifier,
        metrics_config,
        snr,
        testLoader,
        drone_classes,
        device,
        plot_dir=plot_dir,
    )
