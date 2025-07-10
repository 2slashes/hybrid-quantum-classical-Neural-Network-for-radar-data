import numpy as np
import torch
import yaml
from warnings import warn
import os


def dataloader(file_extension):
    data = np.load(file_extension)
    return data


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def load_yaml(yaml_path: str) -> tuple[dict, dict, list]:
    with open(yaml_path, "r") as file:
        user_config_data = yaml.safe_load(file)

        if "modelConfig" not in user_config_data:
            raise Exception('"modelConfig" not present in config file.')
        if "paths" not in user_config_data:
            raise Exception('"paths" not specified in config file.')
        if "classes" not in user_config_data:
            raise Exception('"classes" not specified in config file')

        model_config: dict[str, any] = user_config_data["modelConfig"]
        paths: dict[str, str] = user_config_data["paths"]
        drone_classes: list[str] = user_config_data["classes"]

        if len(drone_classes):
            model_config["num_outputs"] = len(drone_classes)
        else:
            raise Exception(
                '"classes" is empty. Please specify the classes for the model.'
            )

        validate_model_config(model_config)
        validate_paths(paths)

        return model_config, paths, drone_classes


def validate_model_config(model_config):
    if "f_s" not in model_config:
        raise Exception('"f_s" not specified in modelConfig.')
    elif not isinstance(model_config["f_s"], int):
        raise TypeError('"f_s" is not an integer.')

    if "snrList" not in model_config:
        raise Exception('"snrList" not specified in modelConfig')
    elif not isinstance(model_config["snrList"], list):
        raise TypeError('"f_s" is not an list.')
    elif not all(isinstance(entry, int) for entry in model_config["snrList"]):
        raise TypeError('all elements in "snrList" must be integers')

    if "batch_size" not in model_config:
        raise Exception('"batch_size" not specified in modelConfig')
    elif not isinstance(model_config["batch_size"], int):
        raise TypeError('"batch_size" is not an integer.')

    if "epochs" not in model_config:
        raise Exception('"epochs" not specified in modelConfig')
    elif not isinstance(model_config["epochs"], int):
        raise TypeError('"epochs" is not an integer.')

    if "learning_rate" not in model_config:
        raise Exception('"learning_rate" not specified in modelConfig')
    elif not isinstance(model_config["learning_rate"], float):
        raise TypeError('"learning_rate" is not a float.')

    if "model_type" not in model_config:
        raise Exception(
            '"model_type" not specified in modelConfig. Please specify detection or classification.'
        )
    elif not isinstance(model_config["model_type"], str):
        raise TypeError('"model_type" must be a string.')
    model_config["model_type"] = model_config["model_type"].lower()
    if model_config["model_type"] != "detection" and model_config["model_type"] != "classification":
        raise Exception('"model_type" must be detection or classification.')

    if "min_epochs" not in model_config:
        model_config["disable_min_epochs"] = True
    elif not isinstance(model_config["min_epochs"], int):
        raise TypeError('"min_epochs" is not an integer.')
    elif "loss_threshold" not in model_config:
        raise Exception(
            '"min_epochs" specified, but "loss_threshold" is missing from modelConfig.'
        )
    else:
        model_config["disable_min_epochs"] = False

    if "save_model" not in model_config:
        warn('"save_model" not specified in modelConfig. Using False by default.')
        model_config["save_model"] = False
    elif not isinstance(model_config["save_model"], bool):
        raise TypeError('"save_model" is not a boolean.')
    elif model_config["save_model"] and "model_name" not in model_config:
        raise Exception(
            '"save_model" set to true, but "model_name" not specified in modelConfig. Please specify a model name.'
        )

    if "plot_confusion" not in model_config:
        model_config["plot_confusion"] = False
    elif not isinstance(model_config["plot_confusion"], bool):
        raise TypeError('"plot_confusion" is not a boolean.')


def validate_paths(paths):
    if "train_data_dir" not in paths:
        raise Exception('"train_data_dir" not in paths.')
    if not (os.path.isdir(paths["train_data_dir"])):
        raise Exception(f"The directory {paths['train_data_dir']} doesn't exist.")
    if "test_data_dir" not in paths:
        raise Exception('"test_data_dir" not in paths.')
    if not (os.path.isdir(paths["test_data_dir"])):
        raise Exception(f"The directory {paths['test_data_dir']} doesn't exist.")
    if "model_dir" not in paths:
        raise Exception('"model_dir" not in paths.')
    if "plot_dir" not in paths:
        raise Exception('"plot_dir" not in paths.')


def get_num_qlayers_from_yaml(yaml_path: str) -> int:
    with open(yaml_path, "r") as file:
        user_config_data: dict[str, any] = yaml.safe_load(file)
        if "num_qlayers" not in user_config_data:
            raise Exception("num_qlayers must be specified to use hybrid models")
        num_qlayers: int = user_config_data["num_qlayers"]
        return num_qlayers
