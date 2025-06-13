import torch
import torch.nn as nn
from itertools import cycle
from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from torcheval.metrics.functional import multiclass_f1_score

# ---------------------------STUFF-------------------------------------

roc_curve_minimum = 1e-3
drone_type_map = [
    "DJI_Matrice_300_RTK",
    "DJI_Mavic_Air_2",
    "DJI_Mavic_Mini",
    "DJI_Phantom_4",
    "Parrot_Disco",
]
confm_class_map = ["Drones", "Noise"]
model_snr = 5

# ---------------------------GENERAL---------------------------------------

device = None


def get_device():
    global device
    if device is None:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    return device


def dataloader(file_extension):
    data = np.load(file_extension)
    return data


# ---------------------------PLOTTING--------------------------------------


def plot_spectrogram(sp):
    fig, axs = plt.subplots(2, ncols=1, figsize=(14, 3))

    axs[0].imshow(sp[0])
    axs[1].imshow(sp[1])

    plt.tight_layout()
    plt.show()


def plot_multiclass_roc(
    target, probs, snr, plot_file=None, remove_zeros=False, semilog_axes=True
):
    lb = LabelBinarizer().fit(target)
    one_hot_target = lb.transform(target)

    fig, ax = plt.subplots(figsize=(5, 5))
    colors = cycle(["#348ABD", "#b74331", "#8EBA42", "#FBC15E", "#988ED5"])
    for class_id, color in zip(range(5), colors):
        fpr, tpr, _ = roc_curve(one_hot_target[:, class_id], probs[:, class_id])
        if remove_zeros is True:
            zero_indices = np.where(np.isclose(fpr, 0))
            fpr = np.delete(fpr, zero_indices)
            tpr = np.delete(tpr, zero_indices)
        else:
            fpr[np.isclose(fpr, 0)] = 1e-10
        display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(
            name=f"{drone_type_map[class_id]}", ax=ax, color=color
        )
    if semilog_axes is True:
        display.ax_.set_xscale("log")
        display.ax_.set_xlim(roc_curve_minimum, 1.0)
    display.ax_.set_xlabel("False Positive Rate")
    display.ax_.set_ylabel("True Positive Rate")
    line = plt.plot(
        np.geomspace(roc_curve_minimum, 1.0),
        np.geomspace(roc_curve_minimum, 1.0),
        color="k",
        linestyle="dashed",
        label="Chance",
    )
    display.ax_.set_title(f"ROC curves for SNR {snr}dB")
    display.ax_.legend()
    plt.tight_layout()
    if plot_file is not None:
        plt.savefig(plot_file)
    else:
        plt.show()


def plot_sklearn_roc_curve(
    y_real, y_pred, snr, plot_file=None, pos_label=None, remove_zeros=False
):
    fpr, tpr, _ = roc_curve(y_real, y_pred, pos_label=pos_label)
    if remove_zeros is True:
        zero_indices = np.where(np.isclose(fpr, 0))
        fpr = np.delete(fpr, zero_indices)
        tpr = np.delete(tpr, zero_indices)
    else:
        fpr[np.isclose(fpr, 0)] = 1e-5
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=None).plot()
    roc_display.figure_.set_size_inches(5, 5)
    roc_display.ax_.set_xscale("log")
    roc_display.ax_.set_xlim(roc_curve_minimum, 1.0)
    roc_display.ax_.set_xlabel("False Positive Rate")
    line = plt.plot(
        np.geomspace(roc_curve_minimum, 1.0),
        np.geomspace(roc_curve_minimum, 1.0),
        color="g",
    )
    line[0].set_label("Chance")
    roc_display.line_.set_label("Hybrid detector")
    roc_display.ax_.legend()
    plt.tight_layout()
    if plot_file is not None:
        plt.savefig(plot_file)
    else:
        plt.show()


# --------------------------------TRAIN AND TEST----------------------------------------


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


def test(
    conf, net, snr, cur_model_path, testLoader, plot_dir=None, pos_label=None
):
    global device
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
            target.cpu(), probabilities.cpu(), snr, plot_file=roc_plot_file
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
                class_names=drone_type_map,
            )
        else:
            fig, ax = plot_confusion_matrix(
                conf_mat=confm,
                show_normed=True,
                colorbar=True,
                class_names=confm_class_map,
            )
        ax.set_title(f"Confusion matrix for SNR {snr}dB")
        plt.tight_layout()
        if conf_plot_file is not None:
            plt.savefig(conf_plot_file)
        else:
            plt.show()

    f1 = multiclass_f1_score(predicted, target, num_classes=5, average="micro")
    print(f"{f1=}")
