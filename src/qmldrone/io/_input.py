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


def load_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

        if "modelConfig" not in data:
            raise Exception('"modelConfig" not present in config file.')
        if "paths" not in data:
            raise Exception('"paths" not specified in config file.')
        if "classes" not in data:
            raise Exception('"classes" not specified in config file')

        conf = data["modelConfig"]
        paths = data["paths"]
        classes = data["classes"]

        if len(classes):
            conf["num_outputs"] = len(classes)
        else:
            raise Exception(
                '"classes" is empty. Please specify the classes for the model.'
            )

        validate_conf(conf)
        validate_paths(paths)

        return conf, paths, classes


def validate_conf(conf):
    if "f_s" not in conf:
        raise Exception('"f_s" not specified in modelConfig.')
    elif not isinstance(conf["f_s"], int):
        raise TypeError('"f_s" is not an integer.')

    if "snr" not in conf:
        raise Exception('"snr" not specified in modelConfig')
    elif not isinstance(conf["snr"], list):
        raise TypeError('"f_s" is not an list.')
    elif not all(isinstance(entry, int) for entry in conf["snr"]):
        raise TypeError('"all elements in snr must be integers')

    if "batch_size" not in conf:
        raise Exception('"batch_size" not specified in modelConfig')
    elif not isinstance(conf["batch_size"], int):
        raise TypeError('"batch_size" is not an integer.')

    if "epochs" not in conf:
        raise Exception('"epochs" not specified in modelConfig')
    elif not isinstance(conf["epochs"], int):
        raise TypeError('"epochs" is not an integer.')

    if "learning_rate" not in conf:
        raise Exception('"learning_rate" not specified in modelConfig')
    elif not isinstance(conf["learning_rate"], float):
        raise TypeError('"learning_rate" is not a float.')

    if "model_type" not in conf:
        raise Exception(
            '"model_type" not specified in modelConfig. Please specify detection or classification.'
        )
    elif not isinstance(conf["model_type"], str):
        raise TypeError('"model_type" must be a string.')
    conf["model_type"] = conf["model_type"].lower()
    if conf["model_type"] != "detection" and conf["model_type"] != "classification":
        raise Exception('"model_type" must be detection or classification.')
    

    if "min_epochs" not in conf:
        conf["disable_min_epochs"] = True
    elif not isinstance(conf["min_epochs"], int):
        raise TypeError('"min_epochs" is not an integer.')
    elif "loss_threshold" not in conf:
        raise Exception(
            '"min_epochs" specified, but "loss_threshold" is missing from modelConfig.'
        )
    else:
        conf["disable_min_epochs"] = False

    if "save_model" not in conf:
        warn('"save_model" not specified in modelConfig. Using False by default.')
        conf["save_model"] = False
    elif not isinstance(conf["save_model"], bool):
        raise TypeError('"save_model" is not a boolean.')
    elif conf["save_model"] and "model_name" not in conf:
        raise Exception(
            '"save_model" set to true, but "model_name" not specified in modelConfig. Please specify a model name.'
        )
    elif not conf["save_model"]:
        # to be removed once the model doesn't have to be saved between train and test
        conf["model_name"] = "no_save"

    if "plot_confusion" not in conf:
        conf["plot_confusion"] = False
    elif not isinstance(conf["plot_confusion"], bool):
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


def get_num_qlayers_from_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
        if "num_qlayers" not in data:
            raise Exception("num_qlayers must be specified to use hybrid models")
        return data["num_qlayers"]
