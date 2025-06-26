import torch
import torch.nn as nn
from ..visualization.base import plot_base


def test_metrics(
    conf,
    net,
    snr,
    cur_model_path,
    testLoader,
    classes,
    device,
    plot_dir=None,
    pos_label=0,
    model_snr=5,
):
    net = net.to(device)
    net.load_state_dict(torch.load(cur_model_path))
    net.eval()

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

        outputs = net(inputs.float())
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
        conf,
        classes,
        target,
        probabilities,
        predicted,
        pos_label,
    )
