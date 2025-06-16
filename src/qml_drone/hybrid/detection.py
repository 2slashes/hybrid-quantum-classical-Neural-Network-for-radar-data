from qml_drone.base.hybrid import train, test
from qml_drone.base.common import set_outputs, set_model_type


def create():
    set_model_type("detection")
    set_outputs(2)
    train()
    test()
