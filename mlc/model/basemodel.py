from torch import nn


class BaseModel(nn.Module):
    _name = "base_model"

    def __init__(self, args):
        super().__init__()
        self._args = args

    @classmethod
    def name(cls):
        return cls._name

    def args(self):
        return self._args

    @staticmethod
    def add_arguments(parser):
        raise NotImplementedError("BaseModel: Subclasses must implement add_arguments method")
