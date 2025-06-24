import yaml

# ---------------------FILE PATHS-------------------------


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
conf["num_qlayers"] = 4  # number of quantum layers, not used in classical models


def get_conf():
    global conf
    return conf.copy()


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
        if (
            not isinstance(user_conf["num_outputs"], int)
            or user_conf["num_outputs"] <= 0
        ):
            raise Exception("num_outputs must be a positive integer.")
    else:
        user_conf["num_outputs"] = num_classes
    if not isinstance(user_conf["snr"], list) or len(user_conf["snr"]) == 0:
        raise Exception("snr must be a non-empty list.")
    if not all(isinstance(snr, int) for snr in user_conf["snr"]):
        raise Exception(
            "All SNR values must be integers."
        )  # unsure if the SNR values can be floats
    if not isinstance(user_conf["f_s"], int):
        raise Exception("f_s must be an integer.")  # not sure what f_s is
    if not isinstance(user_conf["batch_size"], int) or user_conf["batch_size"] <= 0:
        raise Exception("batch_size must be a positive integer.")
    if not isinstance(user_conf["epochs"], int) or user_conf["epochs"] <= 0:
        raise Exception("epochs must be a positive integer.")
    if not isinstance(user_conf["min_epochs"], int) or user_conf["min_epochs"] <= 0:
        raise Exception("min_epochs must be a positive integer.")
    if (
        not isinstance(user_conf["loss_threshold"], float)
        or user_conf["loss_threshold"] <= 0
    ):
        raise Exception("loss_threshold must be a positive float.")
    if (
        not isinstance(user_conf["learning_rate"], float)
        or user_conf["learning_rate"] <= 0
    ):
        raise Exception("learning_rate must be a positive float.")
    if not isinstance(user_conf["save_model"], bool):
        raise Exception("save_model must be a boolean.")
    if (
        not isinstance(user_conf["model_name"], str)
        or len(user_conf["model_name"]) == 0
    ):
        raise Exception("model_name must be a non-empty string.")
    if not isinstance(user_conf["plot_confusion"], bool):
        raise Exception("plot_confusion must be a boolean.")
    if (
        user_conf["model_type"] != "detection"
        and user_conf["model_type"] != "classification"
    ):
        raise Exception("model_type must be either 'detection' or 'classification'.")
    if (
        "num_qlayers" in user_conf
        and (not isinstance(user_conf["num_qlayers"], int) or user_conf["num_qlayers"])
        <= 0
    ):
        raise Exception("num_qlayers must be a positive integer.")

    # copy only values that are present in 'conf' already
    for key in conf:
        if key in user_conf:
            conf[key] = user_conf[key]
    # manually add num_outputs as the number of classes
    if user_conf["model_type"] == "detection" and num_classes != 2:
        raise Exception(
            "Detection model must have exactly 2 classes (e.g. Signal or Noise)."
        )
    conf["num_outputs"] = num_classes


def load_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

        set_classes(data["classes"])

        user_conf = data["modelConfig"]
        set_conf(user_conf, len(data["classes"]))

        set_paths(data["paths"], user_conf["snr"])
