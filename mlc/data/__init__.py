import importlib
import inspect
import pkgutil
from pathlib import Path

# import mlc.command.base as base
from .basedataset import BaseDataset

# Load all classes from all modules in this package, that are subclasses of Base
_available_datasets = dict()

base_name = __name__.split(".")[0]  # should be "mlc"
# search locally for packages
for mod_info in pkgutil.iter_modules(__path__):
    if mod_info.ispkg is True:
        # model module
        model_path = Path(mod_info.module_finder.path) / str(mod_info.name)
        # print(model_path)
        for mod_info in pkgutil.iter_modules([model_path]):
            model_folder_name = mod_info.module_finder.path.split("/")[-1]
            module = importlib.import_module(f"{base_name}.data.{model_folder_name}.{mod_info.name}")
            for name, class_type in inspect.getmembers(module, inspect.isclass):
                if issubclass(class_type, BaseDataset) and class_type is not BaseDataset:
                    _available_datasets[class_type.name()] = class_type


def get_available_datasets():
    return _available_datasets
