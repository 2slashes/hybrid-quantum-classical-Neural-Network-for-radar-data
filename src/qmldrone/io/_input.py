import numpy as np
import torch
import yaml

def dataloader(file_extension):
    data = np.load(file_extension)
    return data

device = None


def get_device():
    global device
    if device is None:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    return device

def load_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
        conf = data["modelConfig"]
        paths = data["paths"]
        classes = data["classes"]
        conf["num_outputs"] = len(classes)
        return conf, paths, classes