from qml_drone.base.hybrid import train_model, eval_model, set_outputs, set_root_dir, set_epochs, set_model_type, set_qlayers
import torch.nn as nn
import torch
from torcheval.metrics.functional import binary_f1_score
import pennylane as qml

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# confm_class_map = ["Drones", "Noise"]

def create():
    set_model_type("classification")
    set_outputs(5)
    train_model()
    eval_model()

set_qlayers(5)
set_epochs(1)
set_root_dir("../original/Radar")
create()