import importlib
import inspect
import pkgutil
from pathlib import Path

# import mlc.command.base as base
from .base import Base

# Load all classes from all modules in this package, that are subclasses of Base
_available_commands = list()

mod_path = Path(__file__).parent.parent
path = [
    str(mod_path / "command"),  # system commands
    str(mod_path / "data"),  # Commands for data processing
    str(mod_path / "model"),  # Commands for training models
]

base_name = __name__.split(".")[0]  # should be "mlc"

for mod_info in pkgutil.iter_modules(path):
    # print(__name__, mod_info)
    submodule_name = mod_info.module_finder.path.split("/")[-1]
    module = importlib.import_module(f"{base_name}.{submodule_name}.{mod_info.name}")
    for name, class_type in inspect.getmembers(module, inspect.isclass):
        if issubclass(class_type, Base) and class_type is not Base:
            _available_commands.append(class_type)


def get_available_commands():
    return _available_commands
