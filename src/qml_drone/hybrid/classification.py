from qml_drone.base.hybrid import (
    train,
    test,
    set_outputs,
    set_root_dir,
    set_epochs,
    set_model_type,
    set_qlayers,
)


def create():
    set_model_type("classification")
    set_outputs(5)
    train()
    test()


set_qlayers(5)
set_epochs(1)
set_root_dir("../original/Radar")
create()
