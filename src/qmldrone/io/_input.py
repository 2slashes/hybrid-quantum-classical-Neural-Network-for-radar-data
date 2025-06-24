import numpy as np
import torch

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