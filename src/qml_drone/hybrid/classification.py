from qml_drone.base.hybrid import train_model, eval_model, set_outputs, set_root_dir
import torch.nn as nn
import torch
from torcheval.metrics.functional import binary_f1_score
import pennylane as qml

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# confm_class_map = ["Drones", "Noise"]

def create():
    set_outputs(5)
    train_model()
    eval_model()