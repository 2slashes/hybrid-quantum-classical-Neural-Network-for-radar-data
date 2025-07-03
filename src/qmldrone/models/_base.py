from ..io._input import get_device, load_yaml
from ..jobs._base import train, test


def create_base(conf_path: str, classifier):
    conf, paths, classes = load_yaml(conf_path)
    num_outputs = len(classes)
    device = get_device()
    train_conf = {}
    test_conf = {}
    if conf["save_model"]:
        train_conf = dict(
            (key, conf[key])
            for key in (
                "snr",
                "model_name",
                "f_s",
                "batch_size",
                "learning_rate",
                "epochs",
                "min_epochs",
                "loss_threshold",
                "save_model",
            )
        )

        test_conf = dict(
            (key, conf[key])
            for key in (
                "snr",
                "model_name",
                "f_s",
                "batch_size",
                "model_type",
                "plot_confusion",
            )
        )
    else:
        train_conf = dict(
            (key, conf[key])
            for key in (
                "snr",
                "f_s",
                "batch_size",
                "learning_rate",
                "epochs",
                "min_epochs",
                "loss_threshold",
                "save_model",
            )
        )

        test_conf = dict(
            (key, conf[key])
            for key in (
                "snr",
                "f_s",
                "batch_size",
                "model_type",
                "plot_confusion",
            )
        )

    train(
        classifier,
        train_conf,
        num_outputs,
        device,
        paths["model_dir"],
        paths["train_data_dir"],
        conf["disable_min_epochs"]
    )
    test(
        classifier,
        test_conf,
        num_outputs,
        classes,
        device,
        paths["model_dir"],
        paths["plot_dir"],
        paths["test_data_dir"],
    )
