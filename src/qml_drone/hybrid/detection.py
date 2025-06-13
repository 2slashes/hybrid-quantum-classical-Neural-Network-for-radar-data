from qml_drone.base.hybrid import (
    train,
    test,
    set_qlayers,
)
from qml_drone.base.common import (
    set_outputs,
    set_model_type,
    set_epochs,
    set_root_dir
)


def create():
    set_model_type("detection")
    set_outputs(2)
    train()
    test()


set_qlayers(5)
set_epochs(1)
set_root_dir("../original/Radar/binary")
create()
