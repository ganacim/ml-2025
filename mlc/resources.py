from pathlib import Path
import importlib.resources as res 

def data_path() -> Path :
    # get package name
    pkg_name = __name__.split(".")[0]
    # get package path
    with res.path(pkg_name, '.') as path:
        data_path = "/".join(str(path).split("/")[:-1])
    # return data path
    return Path(data_path) / "data"
