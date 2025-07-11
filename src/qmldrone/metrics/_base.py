import torch
import torch.nn as nn
from ..visualization.base import plot_base


def test_metrics(
    classifier,
    model_config: dict[str, any],
    snr: int,
    testLoader,
    drone_classes: list[str],
    device,
    plot_dir: str = None,
    pos_label: int = 0,
    model_snr: int = 5,
) -> None:
    """Part of the test loop for a drone classifier model

    This function evaluates the classifier on the test dataset, computes the loss,
    and collects the predicted labels and probabilities. It then calls the plot_base
    function to visualize the results.

    Args:
        classifier: a RadarClassifier object to test
        model_config: dictionary containing the user's configuration options for the model
        snr: signal to noise ratio of the data
        testLoader: DataLoader object for the test dataset
        drone_classes: list of classes contained in the data
        device: hardware to run the test loop on
        plot_dir: file path to save the plots

    Returns: None
    """
    classifier.eval()

    testloss = 0
    predicted = None
    probabilities = None
    target = None

    loss_fn = nn.CrossEntropyLoss().to(device)

    for i, data in enumerate(testLoader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        if target is None:
            target = labels
        else:
            target = torch.cat((target, labels))

        outputs = classifier(inputs.float())
        loss = loss_fn(outputs, labels)
        testloss += loss.item()

        partial_probabilities = nn.Softmax(dim=1)(outputs.data)
        if probabilities is None:
            probabilities = partial_probabilities
        else:
            probabilities = torch.cat((probabilities, partial_probabilities))

        _, partial_predicted = torch.max(outputs.data, 1)
        if predicted is None:
            predicted = partial_predicted
        else:
            predicted = torch.cat((predicted, partial_predicted))

    target = target.cpu()
    probabilities = probabilities.cpu()
    predicted = predicted.cpu()

    plot_base(
        plot_dir,
        model_snr,
        snr,
        model_config,
        drone_classes,
        target,
        probabilities,
        predicted,
        pos_label,
    )
