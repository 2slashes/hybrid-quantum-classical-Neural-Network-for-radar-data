# ---------------------FILE PATHS-------------------------

paths = {
    "radar_path": "Radar",
    "data_dir": "two_sided",
    "model_dir": "two_sided_models",
    "plot_dir": "two_sided_plots",
}


def set_root_dir(path):
    global paths
    paths["radar_path"] = path
    paths["model_path"] = f"{paths['radar_path']}/{paths['model_dir']}"
    paths["plot_path"] = f"{paths['radar_path']}/{paths['plot_dir']}"


def get_paths():
    global paths
    return paths.copy()


# --------------------DRONE STUFF--------------------------

drone_type_map = [
    "DJI_Matrice_300_RTK",
    "DJI_Mavic_Air_2",
    "DJI_Mavic_Mini",
    "DJI_Phantom_4",
    "Parrot_Disco",
]
confm_class_map = ["Drones", "Noise"]
model_snr = 5


def get_drone_type_map():
    global drone_type_map
    return drone_type_map.copy()


def get_confm_class_map():
    global confm_class_map
    return confm_class_map


def get_model_snr():
    global model_snr
    return model_snr


# -------------------MODEL CONFIG-----------------------

conf = {}
conf["f_s"] = 10_000
conf["SNR"] = [20, 15, 10, 5, 0, -5]
conf["batch_size"] = 8
conf["epochs"] = 200
conf["min_epochs"] = 50
conf["learning_rate"] = 0.001  # these should be parameters
conf["save_model"] = True
conf["model_name"] = "hybrid-parallel-model"
conf["plot_confusion"] = True
conf["model_type"] = "classification"


def set_f_s(f_s: int):
    global conf
    conf["f_s"] = f_s


def set_snr(snr: list[int]):
    global conf
    conf["SNR"] = snr


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


num_outputs: int = None


def set_outputs(outputs: int):
    global num_outputs
    num_outputs = outputs


def get_outputs():
    global num_outputs
    if num_outputs is None:
        raise Exception(
            "Number of outputs not configured. Use set_outputs to set the number of outputs."
        )
    return num_outputs
