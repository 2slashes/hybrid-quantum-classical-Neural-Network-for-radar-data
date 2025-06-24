from itertools import cycle
from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt

def plot_spectrogram(sp):
    fig, axs = plt.subplots(2, ncols=1, figsize=(14, 3))

    axs[0].imshow(sp[0])
    axs[1].imshow(sp[1])

    plt.tight_layout()
    plt.show()


def plot_multiclass_roc(
    target, probs, snr, classes, plot_file=None, remove_zeros=False, semilog_axes=True, roc_curve_minimum=1e-3
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
            name=f"{classes[class_id]}", ax=ax, color=color
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
    y_real, y_pred, snr, plot_file=None, pos_label=None, remove_zeros=False, roc_curve_minimum=1e-3
):
    fpr, tpr, _ = roc_curve(y_real, y_pred, pos_label=pos_label)
    if remove_zeros is True:
        zero_indices = np.where(np.isclose(fpr, 0))
        fpr = np.delete(fpr, zero_indices)
        tpr = np.delete(tpr, zero_indices)
    else:
        fpr[np.isclose(fpr, 0)] = 1e-5
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, name=None).plot()
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
