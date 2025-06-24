from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
from torcheval.metrics.functional import multiclass_f1_score

from qmldrone.visualization.plot import plot_multiclass_roc, plot_sklearn_roc_curve
from ..io._input import get_device

def train(conf, net, model_path, trainLoader):
    device = get_device()
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
        print(f"Saving model state to {model_path}")
        torch.save(net.state_dict(), model_path)


def test(conf, net, snr, cur_model_path, testLoader, classes, plot_dir=None, pos_label=None, model_snr=5):
    device = get_device()
    net = net.to(device)
    net.load_state_dict(torch.load(cur_model_path))
    net.eval()

    correct = 0
    total = 0
    testloss = 0
    predicted = None
    probabilities = None
    target = None

    loss_fn = nn.CrossEntropyLoss().to(device)


    confm = np.zeros((net.outputs, net.outputs), dtype=int)
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
    roc_plot_file = None
    conf_plot_file = None

    if conf["model_type"] == "classification":
        if plot_dir is not None:
            roc_plot_file = (
                f"{plot_dir}/hybrid_classifier_roc_model-{model_snr}_signal-{snr}.pdf"
            )
            conf_plot_file = (
                f"{plot_dir}/hybrid_classifier_conf_model-{model_snr}_signal-{snr}.pdf"
            )
        plot_multiclass_roc(
            target.cpu(), probabilities.cpu(), snr, classes, plot_file=roc_plot_file
        )
    else:
        if pos_label is None:
            pos_label = 0
        plot_file = None
        if plot_dir is not None:
            plot_file = (
                f"{plot_dir}/hybrid_detector_roc_model-{model_snr}_signal-{snr}.pdf"
            )
        plot_sklearn_roc_curve(
            target,
            probabilities[:, pos_label],
            snr,
            plot_file=plot_file,
            pos_label=pos_label,
        )
    confm = confusion_matrix(target, predicted)
    print(f"{confm=}")
    if conf["plot_confusion"] is True:
        plt.close()
        if conf["model_type"] == "classification":
            fig, ax = plot_confusion_matrix(
                conf_mat=confm,
                show_normed=True,
                colorbar=True,
                class_names=classes,
            )
        else:
            fig, ax = plot_confusion_matrix(
                conf_mat=confm,
                show_normed=True,
                colorbar=True,
                class_names=classes,
            )
        ax.set_title(f"Confusion matrix for SNR {snr}dB")
        plt.tight_layout()
        if conf_plot_file is not None:
            plt.savefig(conf_plot_file)
        else:
            plt.show()

    f1 = multiclass_f1_score(predicted, target, num_classes=5, average="micro")
    print(f"{f1=}")
