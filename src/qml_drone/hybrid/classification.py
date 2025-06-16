from qml_drone.base.hybrid import train, test
from qml_drone.base.common import set_outputs, set_model_type


def create():
    set_model_type("classification")
    set_outputs(5)
    train()
    test()
