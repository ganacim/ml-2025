import pkgutil
import importlib
import inspect

# import mlc.command.base as base
from .base import Base

# Load all classes from all modules in this package, that are subclasses of Base
_available_commands = list()
for mod_info in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f"{__name__}.{mod_info.name}")
    for name, class_type in inspect.getmembers(module, inspect.isclass):
        if issubclass(class_type, Base) and class_type is not Base:
            _available_commands.append(class_type)

def get_available_commands():
    return _available_commands