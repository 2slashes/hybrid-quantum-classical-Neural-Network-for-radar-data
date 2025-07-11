from qmldrone.visualization.plot import plot_multiclass_roc


def plot_classification(plot_dir, model_snr, snr, classes, probabilities, target):
    plot_file = None
    if plot_dir is not None:
        plot_file = (
            f"{plot_dir}/hybrid_classifier_roc_model-{model_snr}_signal-{snr}.pdf"
        )

    plot_multiclass_roc(
        target.cpu(), probabilities.cpu(), snr, classes, plot_file=plot_file
    )