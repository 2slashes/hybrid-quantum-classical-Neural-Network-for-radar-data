from ..io._input import get_device, load_yaml
from ..jobs._base import train, test


def create_base(conf_path: str, classifier):
    conf, paths, classes = load_yaml(conf_path)
    device = get_device()
    train(classifier, conf, device, paths["model_dir"], paths["train_data_dir"])
    test(
        classifier,
        conf,
        classes,
        device,
        paths["model_dir"],
        paths["plot_dir"],
        paths["test_data_dir"],
    )
