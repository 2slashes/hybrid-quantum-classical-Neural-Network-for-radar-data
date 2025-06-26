from qmldrone.visualization.plot import plot_sklearn_roc_curve


def plot_detection(plot_dir, model_snr, snr, probabilities, target, pos_label=0):
    plot_file = None
    if plot_dir is not None:
        plot_file = f"{plot_dir}/hybrid_detector_roc_model-{model_snr}_signal-{snr}.pdf"
    plot_sklearn_roc_curve(
        target,
        probabilities[:, pos_label],
        snr,
        plot_file=plot_file,
        pos_label=pos_label,
    )
