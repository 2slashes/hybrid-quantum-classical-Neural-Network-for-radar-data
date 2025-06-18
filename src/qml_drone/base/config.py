import json

# ---------------------FILE PATHS-------------------------

paths = {
    "train_data_dir": "../original/Radar/two_sided/trainset",
    "test_data_dir": "../original/Radar/two_sided/testset",
    "model_dir": "../original/Radartwo_sided_models",
    "plot_dir": "../original/Radartwo_sided_plots",
}


def set_paths(user_paths, snr_list):
    if not isinstance(user_paths, dict):
        raise Exception("Paths must be a dictionary.")
    if not all(
        key in user_paths
        for key in ["train_data_dir", "test_data_dir", "model_dir", "plot_dir"]
    ):
        raise Exception(
            "Paths requires 'train_data_dir', 'test_data_dir', 'model_dir' and 'plot_dir'."
        )

    global paths
    # copy only values that are present in 'paths' already
    for key in paths:
        if key in user_paths:
            paths[key] = user_paths[key]


def get_paths():
    global paths
    return paths.copy()


# --------------------DRONE STUFF--------------------------

classes = [
    "DJI_Matrice_300_RTK",
    "DJI_Mavic_Air_2",
    "DJI_Mavic_Mini",
    "DJI_Phantom_4",
    "Parrot_Disco",
]
model_snr = 5


def get_classes():
    global classes
    return classes.copy()


def set_classes(user_classes):
    if not isinstance(user_classes, list):
        raise Exception("Classes must be a list.")
    if not all(isinstance(cls, str) for cls in user_classes):
        raise Exception("All classes must be strings.")
    if len(user_classes) == 0:
        raise Exception("Classes list cannot be empty.")

    global classes
    classes = user_classes.copy()


# def get_confm_class_map():
#     global confm_class_map
#     return confm_class_map


def get_model_snr():
    global model_snr
    return model_snr


# -------------------MODEL CONFIG-----------------------

conf = {}
conf["f_s"] = 10_000
conf["snr"] = [20, 15, 10, 5, 0, -5]
conf["batch_size"] = 8
conf["epochs"] = 200
conf["min_epochs"] = 50
conf["loss_threshold"] = 0.01
conf["learning_rate"] = 0.001
conf["save_model"] = True
conf["model_name"] = "qml-drone-model"
conf["plot_confusion"] = True
conf["model_type"] = "classification"
conf["num_outputs"] = None


def set_f_s(f_s: int):
    global conf
    conf["f_s"] = f_s


def set_snr(snr: list[int]):
    global conf
    conf["snr"] = snr


def set_batch_size(batch_size: int):
    global conf
    conf["batch_size"] = batch_size


def set_epochs(epochs: int):
    global conf
    conf["epochs"] = epochs


def set_min_epochs(min_epochs: int):
    global conf
    conf["min_epochs"] = min_epochs


def set_learning_rate(learning_rate: float):
    conf["learning_rate"] = learning_rate


def set_save_model(save_model: bool):
    conf["save_model"] = save_model


def set_model_name(model_name: str):
    conf["model_name"] = model_name


def plot_confmat(plot: bool):
    conf["plot_confusion"] = plot


def set_model_type(model_type: str):
    global conf, data_dir, model_dir, plot_dir
    model_type_lower = model_type.lower()
    if model_type_lower != "classification" and model_type_lower != "detection":
        raise Exception("Model type is invalid. Must be detection or classification.")
    conf["model_type"] = model_type_lower


def get_conf():
    global conf
    return conf.copy()


def set_outputs(outputs: int):
    global conf
    conf["num_outputs"] = outputs


def get_outputs():
    global conf
    return conf["num_outputs"]


def set_conf(user_conf, num_classes):
    if not isinstance(user_conf, dict):
        raise Exception("Model configuration must be a dictionary.")
    if not all(
        key in user_conf
        for key in [
            "f_s",
            "snr",
            "batch_size",
            "epochs",
            "min_epochs",
            "loss_threshold",
            "learning_rate",
            "save_model",
            "model_name",
            "plot_confusion",
        ]
    ):
        raise Exception(
            "Model configuration requires 'f_s', 'snr', 'batch_size', 'epochs', 'min_epochs', 'loss_threshold', 'learning_rate', 'save_model', 'model_name' and 'plot_confusion'."
        )
    global conf
    if "num_outputs" in user_conf:
        if not isinstance(user_conf["num_outputs"], int) or user_conf["num_outputs"] <= 0:
            raise Exception("num_outputs must be a positive integer.")
    else:
        user_conf["num_outputs"] = num_classes
    if not isinstance(user_conf["snr"], list) or len(user_conf["snr"]) == 0:
        raise Exception("snr must be a non-empty list.")
    if not all(isinstance(snr, int) for snr in user_conf["snr"]):
        raise Exception("All SNR values must be integers.") # unsure if the SNR values can be floats
    if not isinstance(user_conf["f_s"], int):
        raise Exception("f_s must be an integer.") # not sure what f_s is
    if not isinstance(user_conf["batch_size"], int) or user_conf["batch_size"] <= 0:
        raise Exception("batch_size must be a positive integer.")
    if not isinstance(user_conf["epochs"], int) or user_conf["epochs"] <= 0:
        raise Exception("epochs must be a positive integer.")
    if not isinstance(user_conf["min_epochs"], int) or user_conf["min_epochs"] <= 0:
        raise Exception("min_epochs must be a positive integer.")
    if not isinstance(user_conf["loss_threshold"], float) or user_conf["loss_threshold"] <= 0:
        raise Exception("loss_threshold must be a positive float.")
    if not isinstance(user_conf["learning_rate"], float) or user_conf["learning_rate"] <= 0:
        raise Exception("learning_rate must be a positive float.")
    if not isinstance(user_conf["save_model"], bool):
        raise Exception("save_model must be a boolean.")
    if not isinstance(user_conf["model_name"], str) or len(user_conf["model_name"]) == 0:
        raise Exception("model_name must be a non-empty string.")
    
    if not isinstance(user_conf["plot_confusion"], bool):
        raise Exception("plot_confusion must be a boolean.")
    
    # copy only values that are present in 'conf' already
    for key in conf:
        if key in user_conf:
            conf[key] = user_conf[key]
    # manually add num_outputs as the number of classes
    conf["num_outputs"] = num_classes


def load_json(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)

        set_classes(data["classes"])

        user_conf = data["modelConfig"]
        set_conf(user_conf, len(data["classes"]))

        set_paths(data["paths"], user_conf["snr"])
