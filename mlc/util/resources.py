import importlib.resources as res
from pathlib import Path


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
