from ..io._input import get_device, load_yaml
from ..jobs._base import train, test
import os
import torch


def create_base(conf_path: str, classifier: callable) -> None:
    """Creates a drone classifier model to train and test.

    This function reads the configuration from a YAML file, sets up the model,
    and trains and tests the model for each specified SNR (Signal-to-Noise Ratio).

    Args:
        conf_path: file path to the YAML configuration file.
        classifier: a callable that returns a classifier model instance.
    
    Returns: None
    """
    model_config, paths, drone_classes = load_yaml(conf_path)
    num_outputs = len(drone_classes)
    device = get_device()
    train_config, test_config = setup_train_and_test_config(model_config)

    os.system(f"mkdir -p {paths['model_dir']}")
    os.system(f"mkdir -p {paths['plot_dir']}")

    models = {}
    for snr in model_config["snrList"]:
        models[snr] = classifier(num_outputs).to(device)
        train(
            models[snr],
            train_config,
            device,
            snr,
            paths["train_data_dir"],
            model_config["disable_min_epochs"],
        )
        test(
            models[snr],
            test_config,
            drone_classes,
            device,
            snr,
            paths["plot_dir"],
            paths["test_data_dir"],
        )
        if model_config["save_model"] is True:
            cur_model_path = (
                f"{paths['model_dir']}/{model_config['model_name']}-{snr}.pt"
            )
            print(f"Saving model state to {cur_model_path}")
            torch.save(models[snr].state_dict(), cur_model_path)


def setup_train_and_test_config(model_config: dict[str, any]) -> tuple[dict, dict]:
    train_config: dict[str, any] = {}
    test_config: dict[str, any] = {}
    if model_config["save_model"]:
        train_config = dict(
            (key, model_config[key])
            for key in (
                "snrList",
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

        test_config = dict(
            (key, model_config[key])
            for key in (
                "snrList",
                "model_name",
                "f_s",
                "batch_size",
                "model_type",
                "plot_confusion",
            )
        )
    else:
        train_config = dict(
            (key, model_config[key])
            for key in (
                "snrList",
                "f_s",
                "batch_size",
                "learning_rate",
                "epochs",
                "min_epochs",
                "loss_threshold",
                "save_model",
            )
        )

        test_config = dict(
            (key, model_config[key])
            for key in (
                "snrList",
                "f_s",
                "batch_size",
                "model_type",
                "plot_confusion",
            )
        )
    return train_config, test_config
