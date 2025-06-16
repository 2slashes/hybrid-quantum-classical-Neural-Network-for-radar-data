from qml_drone.base.hybrid import train, test
from qml_drone.base.config import set_outputs, set_model_type, set_model_name


def create():
    set_model_name("Hybrid Drone Classifier")
    set_model_type("classification")
    set_outputs(5)
    train()
    test()
