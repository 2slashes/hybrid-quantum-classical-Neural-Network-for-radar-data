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
    """Training loop for a drone classifier model

    Args:
        classifier: a RadarClassifier object to be trained
        model_config: dictionary containing the user's configuration options for the model
        device: hardware to run the train loop on
        snr: signal to noise ratio of the data
        train_data_dir: file path to the training data
        disable_min_epochs: If true, early termination will not be permitted and the
            training loop will always complete all epochs.

    Returns: None
    """
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
    drone_classes: list[str],
    device,
    snr: int,
    plot_dir: str,
    test_data_dir: str,
):
    """Test loop for a drone classifier model

    Args:
        classifier: a RadarClassifier object to test
        model_config: dictionary containing the user's configuration options for the model
        drone_classes: list of classes contained in the data
        device: hardware to run the test loop on
        snr: signal to noise ratio of the data
        plot_dir: file path to save the plots
        test_data_dir: file path to the test data
    
    Returns: None
    """
    testset_root = f"{test_data_dir}/{model_config['f_s']}fs/{snr}SNR"
    testds = ds.DatasetFolder(testset_root, dataloader, extensions=("npy",))
    testLoader = torch.utils.data.DataLoader(
        testds, model_config["batch_size"], shuffle=True, num_workers=2
    )

    print(f"SNR: {snr} dB")

    metrics_config: dict[str, any] = setup_metrics_config(model_config)

    test_metrics(
        classifier,
        metrics_config,
        snr,
        testLoader,
        drone_classes,
        device,
        plot_dir=plot_dir,
    )


def setup_metrics_config(test_config: dict[str, any]) -> dict:
    metrics_config = dict(
        (key, test_config[key])
        for key in (
            "model_type",
            "plot_confusion",
        )
    )
    return metrics_config
