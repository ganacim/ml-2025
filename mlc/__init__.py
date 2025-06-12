import importlib
import inspect
import pkgutil
from pathlib import Path

# import mlc.command.base as base
from .command.base import Base
from .data.basedataset import BaseDataset
from .model.basemodel import BaseModel

# Load all classes from all modules in this package, that are subclasses of Base
_available_commands = dict()


def _get_available_commands():
    return _available_commands


_available_datasets = dict()


def _get_available_datasets():
    return _available_datasets


_available_models = dict()


def _get_available_models():
    return _available_models


mod_path = Path(__file__).parent
root_paths = [
    str(mod_path),  # system commands
]

base_name = __name__.split(".")[0]  # should be "mlc"

import os # use os.sep to ensure compatibility with different OS
sep = os.path.sep

# recursively load all modules
while len(root_paths) > 0:
    path = root_paths.pop()
    for mod_info in pkgutil.iter_modules([path]):
        if mod_info.ispkg:
            # print(__name__, mod_info.module_finder.path)
            root_paths.append(f"{mod_info.module_finder.path}{sep}{mod_info.name}")
        else:
            module_parts = mod_info.module_finder.path.split(sep)
            base_idx = module_parts.index(base_name)
            submodule_name = ".".join(module_parts[base_idx:])

            module = importlib.import_module(f"{submodule_name}.{mod_info.name}")
            for name, class_type in inspect.getmembers(module, inspect.isclass):
                try:
                    if issubclass(class_type, Base) and class_type is not Base:
                        _available_commands[class_type.name()] = class_type
                    elif issubclass(class_type, BaseDataset) and class_type is not BaseDataset:
                        _available_datasets[class_type.name()] = class_type
                    elif issubclass(class_type, BaseModel) and class_type is not BaseModel:
                        _available_models[class_type.name()] = class_type
                except TypeError:
                    # this is not a subclass of Base
                    print(f"{class_type} name should a @classmethod!")
