import importlib.resources as res
from datetime import datetime
from pathlib import Path

# get current time as string
# this is used to create unique folders for each model
_now_str = datetime.now().strftime("%Y%m%d%H%M%S%f")


def data_path(dataset_name=None) -> Path:
    # get package name
    pkg_name = __name__.split(".")[0]
    # get package path
    with res.path(pkg_name, ".") as path:
        data_path = "/".join(str(path).split("/")[:-1])
    # return data path
    if dataset_name:
        return Path(data_path) / "data" / dataset_name
    else:
        return Path(data_path) / "data"


def model_path(model_name=None, use_personal_folder=False) -> Path:
    # get package name
    pkg_name = __name__.split(".")[0]
    # get package path
    with res.path(pkg_name, ".") as path:
        project_path = Path("/".join(str(path).split("/")[:-1]))
    # use set the personal flag to use the personal folder
    if use_personal_folder:
        project_path = project_path / "personal"
    # return model path
    if model_name:
        return project_path / "models" / model_name
    else:
        return project_path / "models"


def get_time_as_str():
    return _now_str
