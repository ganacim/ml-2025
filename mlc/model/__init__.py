import importlib
import inspect
import pkgutil
from pathlib import Path

# import mlc.command.base as base
from .basemodel import BaseModel

# Load all classes from all modules in this package, that are subclasses of Base
_available_models = list()

base_name = __name__.split(".")[0]  # should be "mlc"
# search locally for packages
for mod_info in pkgutil.iter_modules(__path__):
    if mod_info.ispkg is True:
        # model module
        model_path = Path(mod_info.module_finder.path) / str(mod_info.name)
        # print(model_path)
        for mod_info in pkgutil.iter_modules([model_path]):
            model_folder_name = mod_info.module_finder.path.split("/")[-1]
            module = importlib.import_module(f"{base_name}.model.{model_folder_name}.{mod_info.name}")
            for name, class_type in inspect.getmembers(module, inspect.isclass):
                if issubclass(class_type, BaseModel) and class_type is not BaseModel:
                    _available_models.append(class_type)


def get_available_models():
    return _available_models
