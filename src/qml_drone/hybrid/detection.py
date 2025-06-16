from qml_drone.base.hybrid import train, test
from qml_drone.base.config import set_outputs, set_model_type, set_model_name


def create():
    set_model_name("Hybrid Drone Detector")
    set_model_type("detection")
    set_outputs(2)
    train()
    test()
