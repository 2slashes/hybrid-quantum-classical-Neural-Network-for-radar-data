from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from torcheval.metrics.functional import multiclass_f1_score
from .classification import plot_classification
from .detection import plot_detection


def plot_base(
    plot_dir,
    model_snr,
    snr,
    conf,
    classes,
    target,
    probabilities,
    predicted,
    pos_label=0,
):
    confm_plot_file = None
    if plot_dir is not None:
        confm_plot_file = (
            f"{plot_dir}/hybrid_classifier_conf_model-{model_snr}_signal-{snr}.pdf"
        )

    if conf["model_type"] == "classification":
        plot_classification(plot_dir, model_snr, snr, classes, probabilities, target)
    else:
        plot_detection(plot_dir, model_snr, snr, probabilities, target, pos_label)
    confm = confusion_matrix(target, predicted)
    print(f"{confm=}")
    if conf["plot_confusion"] is True:
        plt.close()
        fig, ax = plot_confusion_matrix(
            conf_mat=confm,
            show_normed=True,
            colorbar=True,
            class_names=classes,
        )
        ax.set_title(f"Confusion matrix for SNR {snr}dB")
        plt.tight_layout()
        if confm_plot_file is not None:
            plt.savefig(confm_plot_file)
        else:
            plt.show()

    f1 = multiclass_f1_score(predicted, target, num_classes=5, average="micro")
    print(f"{f1=}")
