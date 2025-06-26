import numpy as np
import torch
import yaml

def dataloader(file_extension):
    data = np.load(file_extension)
    return data


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    return device

def load_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
        conf = data["modelConfig"]
        paths = data["paths"]
        classes = data["classes"]
        conf["num_outputs"] = len(classes)
        return conf, paths, classes

def get_num_qlayers_from_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
        return data["num_qlayers"]
        